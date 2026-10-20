#include "graphics/host_gpu/renderer/pipeline/pipelineCache.h"

#include "common/assert.h"
#include "common/debugCounters.h"
#include "common/emulatorConfig.h"
#include "common/file.h"
#include "common/logging/log.h"
#include "common/profiler.h"
#include "common/threads.h"
#include "graphics/guest_gpu/gpu_format.h"
#include "graphics/guest_gpu/hardwareContext.h"
#include "graphics/host_gpu/renderer/colorRenderTarget.h"
#include "graphics/host_gpu/renderer/debug.h"
#include "graphics/host_gpu/renderer/depthRenderTarget.h"
#include "graphics/host_gpu/renderer/image/imageView.h"
#include "graphics/host_gpu/renderer/pipeline/blendMapping.h"
#include "graphics/host_gpu/renderer/pipeline/descriptors.h"
#include "graphics/host_gpu/renderer/pipeline/shaderResourceBarrier.h"
#include "graphics/host_gpu/renderer/render.h"
#include "graphics/host_gpu/renderer/renderContext.h"
#include "graphics/presentation/systemOverlay.h"
#include "graphics/shader/recompiler/ShaderRecompiler.h"
#include "graphics/shader/recompiler/ir/ShaderIR.h"
#include "graphics/shader/shaderCompiler.h"
#include "kernel/memory.h"
#include "loader/systemContent.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fmt/format.h>
#include <limits>
#include <mutex>
#include <span>
#include <spirv-tools/libspirv.hpp>
#include <string_view>
#include <utility>
#include <vector>
#include <xxhash.h>

namespace Libs::Graphics {

namespace {

constexpr auto DriverCacheSaveInterval = std::chrono::minutes {5};

uint8_t RemapSourceAlphaFactor(uint8_t factor) {
	switch (static_cast<Prospero::BlendFactor>(factor)) {
		case Prospero::BlendFactor::kSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Alpha);
		case Prospero::BlendFactor::kOneMinusSrcAlpha:
			return static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
		default: return factor;
	}
}

vk::PolygonMode ResolvePolygonMode(const HW::ModeControl& mode, bool cull_front, bool cull_back) {
	// CxPrimitiveSetup::PolygonMode disables both per-face modes when it is zero.
	if (mode.poly_mode == 0) {
		return vk::PolygonMode::eFill;
	}
	EXIT_NOT_IMPLEMENTED(mode.poly_mode != 1);
	if (cull_front && cull_back) {
		return vk::PolygonMode::eFill;
	}
	if (!cull_front && !cull_back && mode.polymode_front_ptype != mode.polymode_back_ptype) {
		EXIT("Pipeline: different polygon modes for two visible faces are unsupported\n");
	}
	// Vulkan has one polygon mode. A culled face does not constrain that mode.
	const auto polygon_mode = cull_front ? mode.polymode_back_ptype : mode.polymode_front_ptype;
	switch (polygon_mode) {
		case 0: return vk::PolygonMode::ePoint;
		case 1: return vk::PolygonMode::eLine;
		case 2: return vk::PolygonMode::eFill;
		default: EXIT("Pipeline: invalid polygon mode %u\n", polygon_mode);
	}
}

std::string DriverCacheSignature(const vk::PhysicalDeviceProperties& properties) {
	constexpr char hex[] = "0123456789abcdef";
	std::string    uuid(VK_UUID_SIZE * 2, '0');
	for (size_t i = 0; i < VK_UUID_SIZE; i++) {
		uuid[i * 2]     = hex[properties.pipelineCacheUUID[i] >> 4u];
		uuid[i * 2 + 1] = hex[properties.pipelineCacheUUID[i] & 0xfu];
	}
	// Driver cache entries already distinguish shader code and pipeline state across builds.
	return fmt::format("KytyPC2:{:08x}:{:08x}:{:08x}:{}\n",
	                   properties.vendorID, properties.deviceID, properties.driverVersion, uuid);
}

std::string PipelineCacheTitleId() {
	std::string title_id;
	if ((!Loader::SystemContentParamSfoGetString("TITLE_ID", &title_id) || title_id.empty()) &&
	    (!Loader::SystemContentParamSfoGetString("CONTENT_ID", &title_id) || title_id.empty())) {
		return {};
	}
	if (!std::ranges::all_of(title_id, [](unsigned char c) {
		    return std::isalnum(c) != 0 || c == '-' || c == '_';
	    })) {
		return {};
	}
	return title_id;
}

template <typename... Args>
void PipelineCacheLog(fmt::format_string<Args...> format, Args&&... args) {
	auto message = fmt::format(format, std::forward<Args>(args)...);
	message += '\n';
	Log::WriteToConsoleAndLog(message);
}

bool ReadShaderGuestMemory(void*, uint64_t address, std::span<uint32_t> values) {
	// Scalar and unformatted buffer dependencies use the same backing as native raw loads.
	// Image synchronization belongs to formatted buffer bindings, not these reads.
	return !values.empty() &&
	       Libs::LibKernel::Memory::TryReadBufferBacking(address, values.data(), values.size_bytes());
}

void DumpShaderSpirv(const char* stage_name, uint64_t shader_hash,
                     const std::vector<uint32_t>& spirv) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / fmt::format("{:04d}_new_shader_{}_{:016x}.spv",
	                                                             id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(spirv.data(), spirv.size() * sizeof(uint32_t));
}

void DumpShaderOriginal(const char* stage_name, uint64_t shader_hash,
                        std::span<const uint32_t> code) {
	if (!Config::GraphicsDebugDumpEnabled()) {
		return;
	}
	EXIT_IF(code.empty());
	static std::atomic_int id = 0;
	const auto path = Config::GetShaderLogFolder() / "original" /
	                  fmt::format("{:04d}_new_shader_{}_{:016x}.bin", id++, stage_name, shader_hash);
	Common::File::CreateDirectories(path.parent_path());
	Common::File file(path);
	if (file.IsInvalid()) {
		const auto path_text = Common::PathToString(path);
		LOGF_COLOR(Log::Color::BrightRed, "Can't create file: %s\n", path_text.c_str());
		return;
	}
	file.Write(code.data(), code.size_bytes());
}

bool ValidateShaderSpirv(const char* label, uint64_t shader_hash,
                         const std::vector<uint32_t>& spirv, bool maintenance8_enabled) {
	if (!Config::ShaderValidationEnabled()) {
		return true;
	}
	spvtools::SpirvTools tools(SPV_ENV_VULKAN_1_3);
	std::string          messages;
	tools.SetMessageConsumer([&messages](spv_message_level_t, const char*,
	                                     const spv_position_t& position, const char* message) {
		messages += fmt::format("{}: {} ({}) {}\n", static_cast<int>(position.line),
		                        static_cast<int>(position.column), static_cast<int>(position.index),
		                        message);
	});
	spvtools::ValidatorOptions validator_options;
	validator_options.SetAllowOffsetTextureOperand(maintenance8_enabled);
	if (tools.Validate(spirv.data(), spirv.size(), validator_options)) {
		return true;
	}
	spvtools::SpirvTools disassembler(SPV_ENV_VULKAN_1_2);
	std::string          text;
	disassembler.Disassemble(spirv, &text,
	                         static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_NO_HEADER) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_FRIENDLY_NAMES) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COMMENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_INDENT) |
	                             static_cast<uint32_t>(SPV_BINARY_TO_TEXT_OPTION_COLOR));
	LOGF_COLOR(Log::Color::BrightRed, "%s SPIR-V validation failed hash=0x%016" PRIx64 ":\n%s",
	           label, shader_hash, messages.c_str());
	LOGF("%s\n", text.c_str());
	return false;
}

} // namespace

std::size_t PipelineCache::GraphicsPipelineKeyHash::operator()(const GraphicsPipelineKey& key) const {
	std::size_t hash = 0;
	PipelineKeyHash::Mix(hash, key.rendering.color_count);
	for (uint32_t i = 0; i < key.rendering.color_count; i++) {
		PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.color_formats[i]));
	}
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.depth_format));
	PipelineKeyHash::Mix(hash, static_cast<uint32_t>(key.rendering.stencil_format));
	for (const auto id: key.vertex_shader_ids) {
		PipelineKeyHash::Mix(hash, id);
	}
	PipelineKeyHash::Mix(hash, key.ps_shader_id);
	PipelineKeyHash::Mix(hash, key.vertex_input.binding_count);
	for (uint32_t i = 0; i < key.vertex_input.binding_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].stride);
		PipelineKeyHash::Mix(hash, key.vertex_input.bindings[i].instance);
	}
	PipelineKeyHash::Mix(hash, key.vertex_input.attribute_count);
	for (uint32_t i = 0; i < key.vertex_input.attribute_count; i++) {
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].offset);
		PipelineKeyHash::Mix(hash, key.vertex_input.attributes[i].binding);
	}
	PipelineKeyHash::Mix(hash, XXH3_64bits(&key.static_params, sizeof(key.static_params)));
	return hash;
}

struct PipelineCache::ProgramCache {
	struct ProgramKey {
		ShaderType            stage           = ShaderType::Unknown;
		uint64_t              hash            = 0;
		uint32_t              user_data_count = 0;
		uint32_t              code_size       = 0;
		std::vector<uint32_t> static_state;

		bool operator==(const ProgramKey&) const = default;
	};

	struct Permutation {
		std::vector<uint32_t>                       spirv;
		std::vector<uint32_t>                       function_code;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		ShaderRecompiler::IR::CompiledShaderInfo     program;
		ShaderProgram                                handle;
	};

	struct SourceEntry {
		SourceEntry(ShaderRecompiler::IR::ResourcePlan plan,
		            std::unique_ptr<ShaderRecompiler::ShaderSource> call_source)
		    : call_source(std::move(call_source)), resource_plan(std::move(plan)) {
			permutations.reserve(8);
		}

		std::span<const uint32_t> FunctionCode() const {
			return call_source
			    ? std::span<const uint32_t>(call_source->linked->code).subspan(call_source->code.size())
			    : std::span<const uint32_t>{};
		}

		bool RefreshCallSource(const ShaderRecompiler::IR::SrtRuntime& runtime) {
			const auto revision = call_source->revision;
			ShaderRecompiler::RefreshShaderSource(*call_source, runtime);
			resource_plan.source_reads = call_source->reads;
			return revision != call_source->revision;
		}

		template <bool HasCalls>
		Permutation* FindPermutation(const ShaderRecompiler::IR::SrtRuntime& runtime,
		                             uint32_t push_data_cursor) {
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    resource_plan, runtime, resources, specialization));
			for (auto& candidate: permutations) {
				const auto& layout = candidate.program.bindings;
				if (layout.push_data_start_dword != ShaderRecompiler::IR::PushData::StartFor(
				        push_data_cursor, layout.ShaderDataDwords()) ||
				    candidate.specialization != specialization) {
					continue;
				}
				if constexpr (HasCalls) {
					if (!std::ranges::equal(candidate.function_code, FunctionCode())) continue;
				}
				return &candidate;
			}
			return nullptr;
		}

		std::unique_ptr<ShaderRecompiler::ShaderSource> call_source;
		ShaderRecompiler::IR::ResourcePlan           resource_plan;
		ShaderRecompiler::IR::ResourceSnapshot       resources;
		ShaderRecompiler::IR::ResourceSpecialization specialization;
		std::vector<Permutation>                    permutations;
	};

	struct ProgramKeyHash {
		std::size_t operator()(const ProgramKey& key) const {
			std::size_t hash = static_cast<std::size_t>(key.stage);
			PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash));
			if constexpr (sizeof(std::size_t) < sizeof(uint64_t)) {
				PipelineKeyHash::Mix(hash, static_cast<std::size_t>(key.hash >> 32u));
			}
			PipelineKeyHash::Mix(hash, key.user_data_count);
			PipelineKeyHash::Mix(hash, key.code_size);
			PipelineKeyHash::Mix(hash, key.static_state.size());
			// Bucket same-shape static variants by source. ProgramKey equality performs the one
			// exact state comparison needed on a stable hit without hashing the full state first.
			return hash;
		}
	};

	static constexpr std::size_t MaxStaticKeyWords = 32 + ShaderVertexInputInfo::RES_MAX * 6;

	Permutation CompilePermutation(const char*                                  stage_name,
	                               const ShaderRecompiler::CompileOptions&      options,
	                               ShaderRecompiler::TranslateResult            translated,
	                               ShaderRecompiler::IR::ResourceSpecialization specialization,
	                               uint32_t push_data_start_dword,
	                               std::span<const uint32_t> function_code) {
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::ShadersCompiled,
		                                         Common::DebugCounters::Counter::ShaderCompileNs,
		                                         Common::DebugCounters::Peak::ShaderCompileNs);
		auto result = ShaderRecompiler::CompileProgram(std::move(translated), options,
		                                               specialization, push_data_start_dword);
		if (!ValidateShaderSpirv(options.dump_label, options.shader_hash, result.spirv,
		                         options.maintenance8_enabled)) {
			DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);
			EXIT("%s failed hash=0x%016" PRIx64 ": SPIR-V validation failed\n", options.dump_label,
			     options.shader_hash);
		}
		DumpShaderSpirv(stage_name, options.shader_hash, result.spirv);

		const auto module = CompileSPV(result.spirv, graphics.device);
		EXIT_IF(module == nullptr);
		if (options.dump_ir) {
			LOGF("%s SPIR-V words=%" PRIu64 " wave_size=%u\n", options.dump_label,
			     static_cast<uint64_t>(result.spirv.size()), options.wave_size);
		}
		return {
		    .spirv          = std::move(result.spirv),
		    .function_code  = {function_code.begin(), function_code.end()},
		    .specialization = std::move(specialization),
		    .program        = std::move(result.program).TakeCompiledInfo(),
		    .handle         = {.id = ++next_shader_id, .module = module},
		};
	}

	template <typename InputInfo>
	ShaderProgram Get(const ShaderParams& params, InputInfo& input_info,
	                  uint32_t& push_data_cursor) {
		ShaderType stage;
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage = input_info.logical_stage;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage = ShaderType::Pixel;
		} else {
			static_assert(std::is_same_v<InputInfo, ShaderComputeInputInfo>);
			stage = ShaderType::Compute;
		}

		const auto user_data = std::span(params.user_data).first(params.user_data_count);
		lookup_key.stage           = stage;
		lookup_key.hash            = params.hash;
		lookup_key.user_data_count = params.user_data_count;
		lookup_key.code_size       = static_cast<uint32_t>(params.code.size());
		BuildStageStaticKey(input_info, lookup_key.static_state);
		auto                                         entry = programs.find(lookup_key);
		ShaderRecompiler::IR::SrtRuntime             runtime {
		    .user_data                  = user_data,
		    .shader_base                = params.Base(),
		    .read_specialization_memory = ReadShaderGuestMemory,
		};
		if constexpr (std::is_same_v<InputInfo, ShaderComputeInputInfo>) {
			runtime.workgroup_counts = input_info.workgroup_counts;
			runtime.indirect_args_address = input_info.indirect_args_address;
		}
		bool source_changed = false;
		if (entry != programs.end()) {
			auto& cached = entry->second;
			Permutation* permutation = nullptr;
			if (cached.call_source) {
				source_changed = cached.RefreshCallSource(runtime);
				if (!source_changed) permutation = cached.FindPermutation<true>(runtime, push_data_cursor);
			} else {
				permutation = cached.FindPermutation<false>(runtime, push_data_cursor);
			}
			if (permutation != nullptr) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}

		ShaderStageInputInfo stage_input {};
		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			stage_input.vertex = &input_info;
		} else if constexpr (std::is_same_v<InputInfo, ShaderPixelInputInfo>) {
			stage_input.pixel = &input_info;
		} else {
			stage_input.compute = &input_info;
		}
		const char* label = nullptr;
		const char* stage_name = nullptr;
		switch (stage) {
			case ShaderType::Vertex: label = "ShaderRecompiler VS"; stage_name = "vs"; break;
			case ShaderType::Mesh: label = "ShaderRecompiler MS"; stage_name = "ms"; break;
			case ShaderType::Local: label = "ShaderRecompiler LS"; stage_name = "ls"; break;
			case ShaderType::TessellationControl: label = "ShaderRecompiler HS"; stage_name = "hs"; break;
			case ShaderType::TessellationEvaluation: label = "ShaderRecompiler DS"; stage_name = "ds"; break;
			case ShaderType::Pixel: label = "ShaderRecompiler PS"; stage_name = "ps"; break;
			case ShaderType::Compute: label = "ShaderRecompiler CS"; stage_name = "cs"; break;
			default: EXIT("invalid pipeline shader stage\n");
		}
		ShaderRecompiler::CompileOptions options;
		options.stage       = stage;
		options.maintenance8_enabled = graphics.maintenance8_enabled;
		options.shader_hash = params.hash;
		options.user_data   = user_data;
		options.back_code      = params.back_code;
		options.dump_ir     = Config::GetShaderLogDirection() != Config::LogDirection::Silent;
		options.early_dump  = options.dump_ir;
		options.emit_debug_info = Config::GraphicsDebugDumpEnabled();
		options.dump_label  = label;
		options.input_info  = stage_input;

		if constexpr (std::is_same_v<InputInfo, ShaderVertexInputInfo>) {
			options.user_data_base = 8;
			options.wave_size = input_info.wave_size;
			if (stage == ShaderType::Mesh || stage == ShaderType::TessellationControl) {
				options.user_data_base = 0;
				options.wave_size = stage == ShaderType::Mesh ? input_info.mesh.wave_size : 64u;
			}
		} else {
			options.wave_size = input_info.wave_size;
		}
		DumpShaderOriginal(stage_name, options.shader_hash, params.code);
		ShaderRecompiler::ShaderSource source;
		if (entry == programs.end()) source = ShaderRecompiler::PrepareShaderSource(params.code, options);
		auto* current = entry == programs.end() ? &source : entry->second.call_source.get();
		ShaderRecompiler::TranslateResult translated;
		if (current != nullptr) {
			if (current->call) {
				if (entry == programs.end()) ShaderRecompiler::RefreshShaderSource(*current, runtime);
				translated = ShaderRecompiler::TranslateProgram(current->linked->decoded, options);
				translated.program.source_reads = current->reads;
			} else {
				translated = ShaderRecompiler::TranslateProgram(current->decoded, options);
			}
		} else {
			translated = ShaderRecompiler::TranslateProgram(params.code, options);
		}
		if (entry == programs.end()) {
			auto retained = source.call ? std::make_unique<ShaderRecompiler::ShaderSource>(std::move(source)) : nullptr;
			entry = programs.try_emplace(lookup_key,
			    ShaderRecompiler::IR::ExtractResourcePlan(translated.program), std::move(retained)).first;
			EXIT_IF(!ShaderRecompiler::IR::MaterializeResources(
			    entry->second.resource_plan, runtime, entry->second.resources,
			    entry->second.specialization));
		} else if (source_changed) {
			entry->second.resource_plan = ShaderRecompiler::IR::ExtractResourcePlan(translated.program);
			if (const auto* permutation = entry->second.FindPermutation<true>(runtime, push_data_cursor)) {
				input_info.stage = {.program = &permutation->program, .resources = &entry->second.resources};
				permutation->program.bindings.AdvancePushData(push_data_cursor);
				return permutation->handle;
			}
		}
		entry->second.permutations.push_back(CompilePermutation(
		    stage_name, options, std::move(translated), entry->second.specialization, push_data_cursor,
		    entry->second.FunctionCode()));
		const auto& permutation = entry->second.permutations.back();
		input_info.stage = {.program = &permutation.program, .resources = &entry->second.resources};
		permutation.program.bindings.AdvancePushData(push_data_cursor);

		std::array<size_t, static_cast<size_t>(ShaderType::TessellationEvaluation) + 1> counts {};
		for (const auto& [key, source]: programs) {
			counts[static_cast<size_t>(key.stage)] += source.permutations.size();
		}
		// Guest geometry shaders are compiled through the host mesh stage.
		std::printf("Shaders: VS %zu | PS %zu | CS %zu | GS %zu | LS %zu | HS %zu | TES %zu\n",
		            counts[static_cast<size_t>(ShaderType::Vertex)],
		            counts[static_cast<size_t>(ShaderType::Pixel)],
		            counts[static_cast<size_t>(ShaderType::Compute)],
		            counts[static_cast<size_t>(ShaderType::Mesh)],
		            counts[static_cast<size_t>(ShaderType::Local)],
		            counts[static_cast<size_t>(ShaderType::TessellationControl)],
		            counts[static_cast<size_t>(ShaderType::TessellationEvaluation)]);
		return permutation.handle;
	}

	explicit ProgramCache(GraphicContext& graphics): graphics(graphics) {
		lookup_key.static_state.reserve(MaxStaticKeyWords);
	}
	~ProgramCache() {
		for (const auto& [key, entry]: programs) {
			(void)key;
			for (const auto& permutation: entry.permutations) {
				graphics.device.destroyShaderModule(permutation.handle.module, nullptr);
			}
		}
	}

	const Permutation* FindPermutation(uint64_t shader_id) const {
		for (const auto& [key, source]: programs) {
			for (const auto& perm: source.permutations) {
				if (perm.handle.id == shader_id) {
					return &perm;
				}
			}
		}
		return nullptr;
	}

	std::unordered_map<ProgramKey, SourceEntry, ProgramKeyHash> programs;
	ProgramKey                                                  lookup_key;
	GraphicContext&                                             graphics;
	uint64_t                                                    next_shader_id = 0;
};

struct ManifestComputeEntry {
	uint64_t                                    shader_id = 0;
	uint32_t                                    wave_size = 64;
	std::vector<vk::DescriptorSetLayoutBinding> bindings;
	std::vector<uint32_t>                       spirv;
};

struct ManifestGraphicsShaderStage {
	ShaderType            stage = ShaderType::Unknown;
	uint32_t              wave_size = 64;
	std::vector<uint32_t> spirv;
};

struct ManifestGraphicsEntry {
	PipelineCache::GraphicsPipelineKey               key {};
	std::vector<ManifestGraphicsShaderStage>         stages;
	std::vector<vk::DescriptorSetLayoutBinding>      bindings;
	std::vector<vk::VertexInputAttributeDescription> input_attributes;
	std::vector<vk::VertexInputBindingDescription>   input_bindings;
	bool                                             mesh = false;
	bool                                             tessellation = false;
	bool                                             rect_list = false;
	uint32_t                                         patch_control_points = 3;
};

static std::unordered_map<uint64_t, ManifestComputeEntry> g_manifest_compute;
static std::unordered_map<PipelineCache::GraphicsPipelineKey, ManifestGraphicsEntry, PipelineCache::GraphicsPipelineKeyHash> g_manifest_graphics;
static std::mutex g_manifest_mutex;

PipelineCache::PipelineCache(GraphicContext& graphics)
    : m_graphics(graphics), m_program_cache(std::make_unique<ProgramCache>(graphics)),
      m_next_driver_cache_save(std::chrono::steady_clock::now() + DriverCacheSaveInterval) {
	EXIT_NOT_IMPLEMENTED(!Common::Thread::IsMainThread());
	InitializeDriverCache();
	LoadManifest();
	PrecompilePipelines();
}

PipelineCache::~PipelineCache() {
	m_precompile_stop.store(true, std::memory_order_release);
	Save();
	auto destroy = [this](const auto& pipelines) {
		for (const auto& [key, pipeline]: pipelines) {
			(void)key;
			m_graphics.device.destroyPipeline(pipeline->pipeline, nullptr);
			m_graphics.device.destroyPipelineLayout(pipeline->pipeline_layout, nullptr);
			m_graphics.device.destroyDescriptorSetLayout(pipeline->descriptor_set_layout, nullptr);
		}
	};
	destroy(m_graphics_pipelines);
	destroy(m_compute_pipelines);
	if (m_driver_cache != nullptr) {
		m_graphics.device.destroyPipelineCache(m_driver_cache, nullptr);
	}
}

void PipelineCache::InitializeDriverCache() {
	const auto title_id = PipelineCacheTitleId();
	if (title_id.empty()) {
		return;
	}
	if (KYTY_BUILD != KYTY_BUILD_RELEASE) {
		PipelineCacheLog("Vulkan pipeline cache: disabled (non-Release build)");
		return;
	}
	m_driver_cache_path     = std::filesystem::path("_PipelineCache") / (title_id + ".bin");
	m_manifest_path         = std::filesystem::path("_PipelineCache") / (title_id + ".manifest");
	const auto path         = Common::PathToString(m_driver_cache_path);
	const bool cache_exists = Common::File::IsFileExisting(m_driver_cache_path);
	if (cache_exists) {
		PipelineCacheLog("Vulkan pipeline cache: loading {}", path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initializing {}", path);
	}
	std::vector<uint8_t> initial_data;
	if (cache_exists) {
		Common::File file(m_driver_cache_path, Common::File::Mode::Read);
		const auto   file_size = file.IsInvalid() ? 0 : file.Size();
		const auto   signature = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
		if (file_size >= signature.size() + sizeof(uint64_t) &&
		    file_size <= std::numeric_limits<uint32_t>::max()) {
			std::string cached_signature(signature.size(), '\0');
			uint64_t    payload_hash = 0;
			initial_data.resize(file_size - signature.size() - sizeof(payload_hash));
			uint32_t signature_read = 0;
			uint32_t hash_read      = 0;
			uint32_t payload_read   = 0;
			file.Read(cached_signature.data(), static_cast<uint32_t>(cached_signature.size()),
			          &signature_read);
			file.Read(&payload_hash, sizeof(payload_hash), &hash_read);
			file.Read(initial_data.data(), static_cast<uint32_t>(initial_data.size()),
			          &payload_read);
			file.Close();
			if (signature_read != cached_signature.size() || hash_read != sizeof(payload_hash) ||
			    payload_read != initial_data.size() || cached_signature != signature ||
			    XXH3_64bits(initial_data.data(), initial_data.size()) != payload_hash) {
				initial_data.clear();
				PipelineCacheLog(
				    "Vulkan pipeline cache: invalidating {} (driver or data mismatch)",
				    path);
			}
		} else {
			file.Close();
			PipelineCacheLog("Vulkan pipeline cache: invalidating {} (invalid file size)", path);
		}
	}

	vk::PipelineCacheCreateInfo create {};
	create.initialDataSize = initial_data.size();
	create.pInitialData    = initial_data.empty() ? nullptr : initial_data.data();
	auto result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	if (result != vk::Result::eSuccess && !initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: driver rejected {} ({}); starting empty", path,
		                 vk::to_string(result));
		initial_data.clear();
		create.initialDataSize = 0;
		create.pInitialData    = nullptr;
		result = m_graphics.device.createPipelineCache(&create, nullptr, &m_driver_cache);
	}
	if (result != vk::Result::eSuccess) {
		PipelineCacheLog("Vulkan pipeline cache: disabled ({})", vk::to_string(result));
		m_driver_cache = nullptr;
		return;
	}
	if (!initial_data.empty()) {
		PipelineCacheLog("Vulkan pipeline cache: loaded {:.2f} MB from {}",
		                 static_cast<double>(initial_data.size()) / 1'000'000.0, path);
	} else {
		PipelineCacheLog("Vulkan pipeline cache: initialized empty");
	}
}

void PipelineCache::CheckpointDriverCache() {
	if (m_driver_cache == nullptr) {
		return;
	}
	m_driver_cache_dirty = true;
	if (std::chrono::steady_clock::now() >= m_next_driver_cache_save) {
		Save();
	}
}

void PipelineCache::Save() {
	SaveManifest();
	if (m_driver_cache == nullptr || !m_driver_cache_dirty) {
		return;
	}
	m_next_driver_cache_save = std::chrono::steady_clock::now() + DriverCacheSaveInterval;

	size_t               size = 0;
	vk::Result           result;
	std::vector<uint8_t> payload;
	for (uint32_t attempt = 0; attempt < 3; attempt++) {
		size   = 0;
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, nullptr);
		if (result != vk::Result::eSuccess || size == 0 ||
		    size > std::numeric_limits<uint32_t>::max()) {
			break;
		}
		payload.resize(size);
		result = m_graphics.device.getPipelineCacheData(m_driver_cache, &size, payload.data());
		if (result != vk::Result::eIncomplete) {
			break;
		}
	}
	if (result != vk::Result::eSuccess || size == 0 ||
	    size > std::numeric_limits<uint32_t>::max()) {
		PipelineCacheLog("Vulkan pipeline cache: save failed ({}, {:.2f} MB)",
		                 vk::to_string(result), static_cast<double>(size) / 1'000'000.0);
		return;
	}
	payload.resize(size);
	auto       prefix       = DriverCacheSignature(m_graphics.GetPhysicalDeviceProperties());
	const auto payload_hash = XXH3_64bits(payload.data(), payload.size());
	prefix.append(reinterpret_cast<const char*>(&payload_hash), sizeof(payload_hash));
	if (!Common::File::CreateDirectories(m_driver_cache_path.parent_path())) {
		PipelineCacheLog("Vulkan pipeline cache: failed to create cache directory");
		return;
	}
	auto temp_path = m_driver_cache_path;
	temp_path += ".tmp";
	Common::File file;
	uint32_t     prefix_written  = 0;
	uint32_t     payload_written = 0;
	if (file.Create(temp_path)) {
		file.Write(prefix.data(), static_cast<uint32_t>(prefix.size()), &prefix_written);
		file.Write(payload.data(), static_cast<uint32_t>(payload.size()), &payload_written);
	}
	const bool flushed = !file.IsInvalid() && file.Flush();
	file.Close();
	const bool written = prefix_written == prefix.size() && payload_written == payload.size() && flushed;
	std::error_code error;
	if (written) {
		std::filesystem::rename(temp_path, m_driver_cache_path, error);
	}
	if (!written || error) {
		PipelineCacheLog("Vulkan pipeline cache: failed to write {}",
		                 Common::PathToString(m_driver_cache_path));
		return;
	}
	PipelineCacheLog("Vulkan pipeline cache: saved {:.2f} MB to {}",
	                 static_cast<double>(payload.size()) / 1'000'000.0,
	                 Common::PathToString(m_driver_cache_path));
	m_driver_cache_dirty = false;
}

void PipelineCache::LoadManifest() {
	if (m_manifest_path.empty() || !Common::File::IsFileExisting(m_manifest_path)) {
		return;
	}
	Common::File file(m_manifest_path, Common::File::Mode::Read);
	if (file.IsInvalid()) {
		return;
	}
	char magic[8] = {};
	uint32_t bytes_read = 0;
	file.Read(magic, 8, &bytes_read);
	if (std::memcmp(magic, "KYTYPIPE", 8) != 0) {
		return;
	}
	uint32_t version = 0, compute_count = 0, graphics_count = 0;
	file.Read(&version, sizeof(version), &bytes_read);
	if (version != 1) return;
	file.Read(&compute_count, sizeof(compute_count), &bytes_read);
	file.Read(&graphics_count, sizeof(graphics_count), &bytes_read);

	std::scoped_lock lock(g_manifest_mutex);
	g_manifest_compute.clear();
	g_manifest_graphics.clear();

	for (uint32_t c = 0; c < compute_count; c++) {
		ManifestComputeEntry entry;
		file.Read(&entry.shader_id, sizeof(entry.shader_id), &bytes_read);
		file.Read(&entry.wave_size, sizeof(entry.wave_size), &bytes_read);
		uint32_t binding_count = 0;
		file.Read(&binding_count, sizeof(binding_count), &bytes_read);
		entry.bindings.resize(binding_count);
		for (uint32_t b = 0; b < binding_count; b++) {
			uint32_t binding = 0, desc_type = 0, desc_count = 0, stage_flags = 0;
			file.Read(&binding, sizeof(binding), &bytes_read);
			file.Read(&desc_type, sizeof(desc_type), &bytes_read);
			file.Read(&desc_count, sizeof(desc_count), &bytes_read);
			file.Read(&stage_flags, sizeof(stage_flags), &bytes_read);
			entry.bindings[b] = {binding, static_cast<vk::DescriptorType>(desc_type),
			                     desc_count, vk::ShaderStageFlags {stage_flags}, nullptr};
		}
		uint32_t spirv_count = 0;
		file.Read(&spirv_count, sizeof(spirv_count), &bytes_read);
		entry.spirv.resize(spirv_count);
		file.Read(entry.spirv.data(), spirv_count * sizeof(uint32_t), &bytes_read);
		g_manifest_compute.emplace(entry.shader_id, std::move(entry));
	}

	for (uint32_t g = 0; g < graphics_count; g++) {
		ManifestGraphicsEntry entry;
		file.Read(&entry.key, sizeof(entry.key), &bytes_read);
		file.Read(&entry.mesh, sizeof(entry.mesh), &bytes_read);
		file.Read(&entry.tessellation, sizeof(entry.tessellation), &bytes_read);
		file.Read(&entry.rect_list, sizeof(entry.rect_list), &bytes_read);
		file.Read(&entry.patch_control_points, sizeof(entry.patch_control_points), &bytes_read);
		uint32_t stage_count = 0;
		file.Read(&stage_count, sizeof(stage_count), &bytes_read);
		entry.stages.resize(stage_count);
		for (uint32_t s = 0; s < stage_count; s++) {
			uint32_t stage = 0, wave_size = 0, spirv_count = 0;
			file.Read(&stage, sizeof(stage), &bytes_read);
			file.Read(&wave_size, sizeof(wave_size), &bytes_read);
			file.Read(&spirv_count, sizeof(spirv_count), &bytes_read);
			entry.stages[s].stage = static_cast<ShaderType>(stage);
			entry.stages[s].wave_size = wave_size;
			entry.stages[s].spirv.resize(spirv_count);
			file.Read(entry.stages[s].spirv.data(), spirv_count * sizeof(uint32_t), &bytes_read);
		}
		uint32_t binding_count = 0;
		file.Read(&binding_count, sizeof(binding_count), &bytes_read);
		entry.bindings.resize(binding_count);
		for (uint32_t b = 0; b < binding_count; b++) {
			uint32_t binding = 0, desc_type = 0, desc_count = 0, stage_flags = 0;
			file.Read(&binding, sizeof(binding), &bytes_read);
			file.Read(&desc_type, sizeof(desc_type), &bytes_read);
			file.Read(&desc_count, sizeof(desc_count), &bytes_read);
			file.Read(&stage_flags, sizeof(stage_flags), &bytes_read);
			entry.bindings[b] = {binding, static_cast<vk::DescriptorType>(desc_type),
			                     desc_count, vk::ShaderStageFlags {stage_flags}, nullptr};
		}
		uint32_t attr_count = 0;
		file.Read(&attr_count, sizeof(attr_count), &bytes_read);
		entry.input_attributes.resize(attr_count);
		for (uint32_t a = 0; a < attr_count; a++) {
			uint32_t location = 0, binding = 0, format = 0, offset = 0;
			file.Read(&location, sizeof(location), &bytes_read);
			file.Read(&binding, sizeof(binding), &bytes_read);
			file.Read(&format, sizeof(format), &bytes_read);
			file.Read(&offset, sizeof(offset), &bytes_read);
			vk::VertexInputAttributeDescription attr_desc {};
			attr_desc.location = location;
			attr_desc.binding  = binding;
			attr_desc.format   = static_cast<vk::Format>(format);
			attr_desc.offset   = offset;
			entry.input_attributes[a] = attr_desc;
		}
		uint32_t bind_count = 0;
		file.Read(&bind_count, sizeof(bind_count), &bytes_read);
		entry.input_bindings.resize(bind_count);
		for (uint32_t bi = 0; bi < bind_count; bi++) {
			uint32_t binding = 0, stride = 0, input_rate = 0;
			file.Read(&binding, sizeof(binding), &bytes_read);
			file.Read(&stride, sizeof(stride), &bytes_read);
			file.Read(&input_rate, sizeof(input_rate), &bytes_read);
			vk::VertexInputBindingDescription bind_desc {};
			bind_desc.binding   = binding;
			bind_desc.stride    = stride;
			bind_desc.inputRate = static_cast<vk::VertexInputRate>(input_rate);
			entry.input_bindings[bi] = bind_desc;
		}
		g_manifest_graphics.emplace(entry.key, std::move(entry));
	}
	PipelineCacheLog("Vulkan pipeline manifest: loaded {} compute and {} graphics pipelines",
	                 compute_count, graphics_count);
}

void PipelineCache::SaveManifest() {
	if (m_manifest_path.empty() || !m_manifest_dirty) {
		return;
	}
	std::scoped_lock lock(g_manifest_mutex);
	auto temp_path = m_manifest_path;
	temp_path += ".tmp";
	Common::File file;
	if (!file.Create(temp_path)) {
		return;
	}
	const char magic[8] = {'K','Y','T','Y','P','I','P','E'};
	file.Write(magic, 8);
	const uint32_t version = 1;
	file.Write(&version, sizeof(version));
	const uint32_t compute_count = static_cast<uint32_t>(g_manifest_compute.size());
	const uint32_t graphics_count = static_cast<uint32_t>(g_manifest_graphics.size());
	file.Write(&compute_count, sizeof(compute_count));
	file.Write(&graphics_count, sizeof(graphics_count));

	for (const auto& [id, entry]: g_manifest_compute) {
		file.Write(&entry.shader_id, sizeof(entry.shader_id));
		file.Write(&entry.wave_size, sizeof(entry.wave_size));
		const uint32_t binding_count = static_cast<uint32_t>(entry.bindings.size());
		file.Write(&binding_count, sizeof(binding_count));
		for (const auto& b: entry.bindings) {
			const uint32_t binding = b.binding;
			const uint32_t desc_type = static_cast<uint32_t>(b.descriptorType);
			const uint32_t desc_count = b.descriptorCount;
			const uint32_t stage_flags = static_cast<uint32_t>(b.stageFlags);
			file.Write(&binding, sizeof(binding));
			file.Write(&desc_type, sizeof(desc_type));
			file.Write(&desc_count, sizeof(desc_count));
			file.Write(&stage_flags, sizeof(stage_flags));
		}
		const uint32_t spirv_count = static_cast<uint32_t>(entry.spirv.size());
		file.Write(&spirv_count, sizeof(spirv_count));
		file.Write(entry.spirv.data(), spirv_count * sizeof(uint32_t));
	}

	for (const auto& [k, entry]: g_manifest_graphics) {
		file.Write(&entry.key, sizeof(entry.key));
		file.Write(&entry.mesh, sizeof(entry.mesh));
		file.Write(&entry.tessellation, sizeof(entry.tessellation));
		file.Write(&entry.rect_list, sizeof(entry.rect_list));
		file.Write(&entry.patch_control_points, sizeof(entry.patch_control_points));
		const uint32_t stage_count = static_cast<uint32_t>(entry.stages.size());
		file.Write(&stage_count, sizeof(stage_count));
		for (const auto& s: entry.stages) {
			const uint32_t stage = static_cast<uint32_t>(s.stage);
			const uint32_t wave_size = s.wave_size;
			const uint32_t spirv_count = static_cast<uint32_t>(s.spirv.size());
			file.Write(&stage, sizeof(stage));
			file.Write(&wave_size, sizeof(wave_size));
			file.Write(&spirv_count, sizeof(spirv_count));
			file.Write(s.spirv.data(), spirv_count * sizeof(uint32_t));
		}
		const uint32_t binding_count = static_cast<uint32_t>(entry.bindings.size());
		file.Write(&binding_count, sizeof(binding_count));
		for (const auto& b: entry.bindings) {
			const uint32_t binding = b.binding;
			const uint32_t desc_type = static_cast<uint32_t>(b.descriptorType);
			const uint32_t desc_count = b.descriptorCount;
			const uint32_t stage_flags = static_cast<uint32_t>(b.stageFlags);
			file.Write(&binding, sizeof(binding));
			file.Write(&desc_type, sizeof(desc_type));
			file.Write(&desc_count, sizeof(desc_count));
			file.Write(&stage_flags, sizeof(stage_flags));
		}
		const uint32_t attr_count = static_cast<uint32_t>(entry.input_attributes.size());
		file.Write(&attr_count, sizeof(attr_count));
		for (const auto& a: entry.input_attributes) {
			const uint32_t location = a.location;
			const uint32_t binding = a.binding;
			const uint32_t format = static_cast<uint32_t>(a.format);
			const uint32_t offset = a.offset;
			file.Write(&location, sizeof(location));
			file.Write(&binding, sizeof(binding));
			file.Write(&format, sizeof(format));
			file.Write(&offset, sizeof(offset));
		}
		const uint32_t bind_count = static_cast<uint32_t>(entry.input_bindings.size());
		file.Write(&bind_count, sizeof(bind_count));
		for (const auto& bi: entry.input_bindings) {
			const uint32_t binding = bi.binding;
			const uint32_t stride = bi.stride;
			const uint32_t input_rate = static_cast<uint32_t>(bi.inputRate);
			file.Write(&binding, sizeof(binding));
			file.Write(&stride, sizeof(stride));
			file.Write(&input_rate, sizeof(input_rate));
		}
	}
	file.Flush();
	file.Close();
	std::error_code error;
	std::filesystem::rename(temp_path, m_manifest_path, error);
	m_manifest_dirty = false;
	PipelineCacheLog("Vulkan pipeline manifest: saved {} pipelines", compute_count + graphics_count);
}

void PipelineCache::PrecompilePipelines() {
	const uint32_t total = static_cast<uint32_t>(g_manifest_compute.size() + g_manifest_graphics.size());
	if (total == 0) {
		SetPipelineCompilationProgress(false, 0, 0);
		return;
	}

	SetPipelineCompilationProgress(true, 0, total);
	Common::Thread precompile_thread([](void* arg) {
		auto* self = static_cast<PipelineCache*>(arg);
		self->PrecompilePipelinesWorker();
	}, this);
	precompile_thread.Detach();
}

void PipelineCache::PrecompilePipelinesWorker() {
	const uint32_t total = static_cast<uint32_t>(g_manifest_compute.size() + g_manifest_graphics.size());
	uint32_t current = 0;

	// 1. Compute Pipelines
	{
		std::scoped_lock lock(g_manifest_mutex);
		for (const auto& [id, entry]: g_manifest_compute) {
			if (m_precompile_stop.load(std::memory_order_relaxed)) {
				break;
			}
			if (m_compute_pipelines.contains(id)) {
				current++;
				SetPipelineCompilationProgress(true, current, total);
				continue;
			}

			auto cached = std::make_unique<Pipeline>();
			vk::ShaderModuleCreateInfo sm_info {};
			sm_info.codeSize = entry.spirv.size() * sizeof(uint32_t);
			sm_info.pCode = entry.spirv.data();
			vk::ShaderModule module = nullptr;
			if (m_graphics.device.createShaderModule(&sm_info, nullptr, &module) == vk::Result::eSuccess) {
				CreateDescriptorLayout(m_graphics, *cached, entry.bindings);
				const vk::PushConstantRange push_constants {vk::ShaderStageFlagBits::eCompute, 0,
				                                            ShaderRecompiler::IR::NativePushConstantSize};
				vk::PipelineLayoutCreateInfo pl_info {};
				pl_info.setLayoutCount = 1;
				pl_info.pSetLayouts = &cached->descriptor_set_layout;
				pl_info.pushConstantRangeCount = 1;
				pl_info.pPushConstantRanges = &push_constants;
				if (m_graphics.device.createPipelineLayout(&pl_info, nullptr, &cached->pipeline_layout) == vk::Result::eSuccess) {
					vk::ComputePipelineCreateInfo cp_info {};
					cp_info.stage.stage = vk::ShaderStageFlagBits::eCompute;
					cp_info.stage.module = module;
					cp_info.stage.pName = "main";
					vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo comp_subgroup_size {};
					if (m_graphics.compute_subgroup_size_control_enabled &&
					    entry.wave_size >= m_graphics.min_subgroup_size && entry.wave_size <= m_graphics.max_subgroup_size) {
						comp_subgroup_size.requiredSubgroupSize = entry.wave_size;
						cp_info.stage.pNext = &comp_subgroup_size;
					}
					cp_info.layout = cached->pipeline_layout;
					cp_info.basePipelineIndex = -1;
					if (m_graphics.device.createComputePipelines(m_driver_cache, 1, &cp_info, nullptr, &cached->pipeline) == vk::Result::eSuccess) {
						m_compute_pipelines.emplace(id, std::move(cached));
					}
				}
				m_graphics.device.destroyShaderModule(module, nullptr);
			}
			current++;
			SetPipelineCompilationProgress(true, current, total);
		}
	}

	// 2. Graphics Pipelines
	{
		std::scoped_lock lock(g_manifest_mutex);
		for (const auto& [key, entry]: g_manifest_graphics) {
			if (m_precompile_stop.load(std::memory_order_relaxed)) {
				break;
			}
			if (m_graphics_pipelines.contains(key)) {
				current++;
				SetPipelineCompilationProgress(true, current, total);
				continue;
			}

			auto cached = std::make_unique<Pipeline>();
			std::vector<vk::ShaderModule> modules_to_destroy;
			std::vector<vk::PipelineShaderStageCreateInfo> shader_stages;
			for (const auto& st: entry.stages) {
				vk::ShaderModuleCreateInfo sm_info {};
				sm_info.codeSize = st.spirv.size() * sizeof(uint32_t);
				sm_info.pCode = st.spirv.data();
				vk::ShaderModule module = nullptr;
				if (m_graphics.device.createShaderModule(&sm_info, nullptr, &module) == vk::Result::eSuccess) {
					modules_to_destroy.push_back(module);
					vk::PipelineShaderStageCreateInfo stage_ci {};
					stage_ci.stage  = NativeShaderStage(st.stage);
					stage_ci.module = module;
					stage_ci.pName  = "main";
					shader_stages.push_back(stage_ci);
				}
			}

			if (shader_stages.size() == entry.stages.size()) {
				CreateDescriptorLayout(m_graphics, *cached, entry.bindings);
				vk::ShaderStageFlags graphics_stages = vk::ShaderStageFlagBits::eFragment;
				for (const auto& st: entry.stages) {
					graphics_stages |= NativeShaderStage(st.stage);
				}
				const vk::PushConstantRange push_constants {graphics_stages, 0,
				                                            ShaderRecompiler::IR::NativePushConstantSize};
				vk::PipelineLayoutCreateInfo pl_info {};
				pl_info.setLayoutCount = 1;
				pl_info.pSetLayouts = &cached->descriptor_set_layout;
				pl_info.pushConstantRangeCount = 1;
				pl_info.pPushConstantRanges = &push_constants;
				if (m_graphics.device.createPipelineLayout(&pl_info, nullptr, &cached->pipeline_layout) == vk::Result::eSuccess) {
					CreatePrecompiledGraphicsPipeline(
						m_graphics, *cached, key.rendering, key.static_params,
						shader_stages, entry.input_attributes, entry.input_bindings,
						entry.mesh, entry.tessellation, entry.rect_list,
						entry.patch_control_points, m_driver_cache);
					if (cached->pipeline != nullptr) {
						m_graphics_pipelines.emplace(key, std::move(cached));
					}
				}
			}

			for (auto m: modules_to_destroy) {
				m_graphics.device.destroyShaderModule(m, nullptr);
			}

			current++;
			SetPipelineCompilationProgress(true, current, total);
		}
	}

	SetPipelineCompilationProgress(false, total, total);
	PipelineCacheLog("Vulkan pipeline precompilation finished: precompiled {} pipelines", total);
}

PipelineCache::GraphicsPrograms PipelineCache::GetGraphicsPrograms(
    const HW::VertexShaderInfo& vertex_regs, const HW::PixelShaderInfo& pixel_regs,
    const HW::ShaderRegisters& sh, const HW::Context& context, const HW::UserConfig& user_config,
    std::span<const Prospero::ColorComponentMapping, 8> target_export_mapping, bool pixel_active,
    std::array<ShaderVertexInputInfo, 3>& vertex_info, ShaderPixelInputInfo& pixel_info) {
	const bool tess_active = user_config.GetPrimType() == Prospero::PrimitiveType::kPatch;
	std::array<ShaderParams, 3> vertex_params;
	if (tess_active) {
		vertex_params = PrepareTessellationPrograms(vertex_regs, context, vertex_info);
	} else {
		vertex_params[0] = PrepareProgram(vertex_regs, context, user_config, vertex_info[0]);
	}
	const bool mesh_active = vertex_info[0].logical_stage == ShaderType::Mesh;
	const bool object_primitive_id = context.HasObjectPrimitiveIdOutput();
	if (object_primitive_id) {
		// CxObjectIdMode: register source plus a 32-bit pipelined primitive-ID addition.
		EXIT_NOT_IMPLEMENTED(sh.m_paClObjprimIdCntl != 2u || mesh_active || tess_active ||
		                     !pixel_active || (sh.m_vgtDrawPayloadCntl & 8u) == 0 ||
		                     user_config.GetPrimType() != Prospero::PrimitiveType::kTriList);
		const auto& target = context.GetRenderTarget(7);
		EXIT_NOT_IMPLEMENTED(Prospero::ResolveRenderTargetFormat(target.info.format,
		                                                       target.info.channel_type)
		                         .buffer_format != Prospero::BufferFormat::k32UInt);
	}
	if (mesh_active) {
		EXIT_NOT_IMPLEMENTED(!m_graphics.mesh_shader_enabled);
		auto& mesh              = vertex_info[0].mesh;
		mesh.host_subgroup_size = m_graphics.subgroup_size;
		const auto& limits      = m_graphics.mesh_shader_properties;
		const auto  logical_threads =
		    mesh.threads_num[0] * mesh.threads_num[1] * mesh.threads_num[2];
		const auto host_threads = ((logical_threads + mesh.wave_size - 1u) / mesh.wave_size) *
		                          std::min(mesh.host_subgroup_size, mesh.wave_size);
		if (host_threads > limits.maxMeshWorkGroupInvocations ||
		    host_threads > limits.maxMeshWorkGroupSize[0] ||
		    mesh.max_vertices > limits.maxMeshOutputVertices ||
		    mesh.max_primitives > limits.maxMeshOutputPrimitives ||
		    mesh.lds_size_dwords * sizeof(uint32_t) > limits.maxMeshSharedMemorySize) {
			EXIT("mesh shader exceeds host limits: threads=%u vertices=%u primitives=%u LDS=%u\n",
			     host_threads, mesh.max_vertices, mesh.max_primitives, mesh.lds_size_dwords);
		}
	}
	ShaderParams pixel_params;
	if (pixel_active) {
		pixel_params      = PrepareProgram(pixel_regs, sh, target_export_mapping, pixel_info);
		pixel_info.object_primitive_id = object_primitive_id;
		for (uint32_t export_index = 0; export_index < 8u; export_index++) {
			const auto slot = ShaderPixelExportTarget(pixel_info.target_shader_mask, export_index);
			if (slot >= 8u || pixel_info.target_output_mode[export_index] == 0u) continue;
			const auto& rt = context.GetRenderTarget(slot);
			const auto target_mask = render_target_mask_slot(context.GetRenderTargetMask(), slot);
			if (rt.base.addr == 0 || target_mask == 0u) continue;
			const auto format = Prospero::ResolveRenderTargetFormat(rt.info.format, rt.info.channel_type);
			if (format.buffer_format != Prospero::BufferFormat::k11_11_10UNorm) continue;
			const auto mapping = pixel_info.target_export_mapping[export_index];
			const auto shader_mask = render_target_mask_slot(pixel_info.target_shader_mask, slot);
			const bool blend = context.GetBlendControl(slot).enable && !rt.info.blend_bypass;
			if (blend || (mapping.ApplyMask(target_mask) & 7u) != 7u ||
			    (mapping.ApplyMask(shader_mask) & 7u) != 7u ||
			    pixel_info.target_output_mode[export_index] != 9u) {
				EXIT("unsupported packed UNorm render-target export: slot=%u export=%u mode=%u "
				     "target_mask=0x%x shader_mask=0x%x mapping=0x%02x blend=%u\n",
				     slot, export_index, pixel_info.target_output_mode[export_index], target_mask,
				     shader_mask, mapping.packed, static_cast<uint32_t>(blend));
			}
			pixel_info.target_packed_unorm_mask |= 1u << export_index;
			pixel_info.target_round_mode |= static_cast<uint8_t>(rt.info.round_mode) << export_index;
		}
		if (!mesh_active && !tess_active && Prospero::IsRectList(user_config.GetPrimType())) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::Rectangle;
		} else if (context.GetModeControl().provoking_vtx_last) {
			pixel_info.parameter_mode = ShaderPixelParameterMode::LastVertex;
		} else {
			pixel_info.parameter_mode = ShaderPixelParameterMode::FirstVertex;
		}
		const auto& blend = context.GetBlendControl(0);
		pixel_info.dual_source_blending =
		    blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		    (BlendFactorIsDualSource(blend.color_srcblend) ||
		     BlendFactorIsDualSource(blend.color_destblend) ||
		     (blend.separate_alpha_blend && (BlendFactorIsDualSource(blend.alpha_srcblend) ||
		                                     BlendFactorIsDualSource(blend.alpha_destblend))));
		if (pixel_info.dual_source_blending) {
			// MRT1 supplies the second blend source for target 0.
			pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
			pixel_info.target_export_mapping[1] = pixel_info.target_export_mapping[0];
		} else if (blend.enable && !context.GetRenderTarget(0).info.blend_bypass &&
		           pixel_info.target_output_mode[0] != 0 && pixel_info.target_output_mode[0] != 7 &&
		           std::all_of(std::begin(pixel_info.target_output_mode) + 1,
		                       std::end(pixel_info.target_output_mode),
		                       [](uint8_t mode) { return mode == 0; })) {
			switch (ClassifyBlendMapping(blend, pixel_info.target_export_mapping[0])) {
				case BlendMappingSupport::SourceAlpha:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlpha;
					break;
				case BlendMappingSupport::SourceAlphaOne:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaOne;
					break;
				case BlendMappingSupport::SourceAlphaZero:
					pixel_info.alpha_blend_source = ShaderAlphaBlendSource::SourceAlphaZero;
					break;
				default: break;
			}
			if (pixel_info.alpha_blend_source != ShaderAlphaBlendSource::None) {
				pixel_info.dual_source_blending     = true;
				pixel_info.target_output_mode[1]    = pixel_info.target_output_mode[0];
				pixel_info.target_export_mapping[1] = {};
			}
		}
	}
	if (context.GetClipControl().clip_disable) {
		const auto& viewport = context.GetScreenViewport().viewports[0];
		const auto& limits   = m_graphics.GetPhysicalDeviceProperties().limits;
		auto&       clip     = vertex_info[tess_active ? 2u : 0u].clip_space;
		clip.scale[0]        = viewport.xscale;
		clip.scale[1]        = viewport.yscale;
		clip.offset[0]       = viewport.xoffset;
		clip.offset[1]       = viewport.yoffset;
		clip.half_extent[0] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[0], 16384u)) * 0.5f;
		clip.half_extent[1] =
		    static_cast<float>(std::min(limits.maxViewportDimensions[1], 16384u)) * 0.5f;
		clip.enabled = true;
	}
	uint32_t push_data_cursor = ShaderRecompiler::IR::PushData::GraphicsDrawDwords(
	    mesh_active, object_primitive_id);
	GraphicsPrograms  result;
	if (pixel_active) {
		result.pixel = m_program_cache->Get(pixel_params, pixel_info, push_data_cursor);
	}
	for (uint32_t i = 0; i < (tess_active ? 3u : 1u); i++) {
		result.vertex[i] = m_program_cache->Get(vertex_params[i], vertex_info[i], push_data_cursor);
	}
	return result;
}

ShaderProgram PipelineCache::GetComputeProgram(const HW::ComputeShaderInfo& regs,
                                               const HW::ShaderRegisters&   sh,
                                               ShaderComputeInputInfo&      input_info) {
	input_info.host_subgroup_size = m_graphics.SupportsComputeWave64() ? 64u : 32u;
	const auto        params      = PrepareProgram(regs, sh, input_info);
	input_info.lds_storage = input_info.lds_size_dwords * 4u >
	    m_graphics.GetPhysicalDeviceProperties().limits.maxComputeSharedMemorySize;
	uint32_t          push_data_cursor = 0;
	return m_program_cache->Get(params, input_info, push_data_cursor);
}

bool PipelineStaticParameters::operator==(const PipelineStaticParameters& other) const noexcept {
	return std::memcmp(this, &other, sizeof(*this)) == 0;
}

PipelineCache::Pipeline& PipelineCache::GetGraphicsPipeline(
    std::span<const RenderColorInfo> colors, const RenderDepthInfo& depth,
    std::span<const ShaderVertexInputInfo> vertex_info, CommandBuffer& command,
    const ShaderPixelInputInfo* ps_input_info, vk::PrimitiveTopology topology,
    bool primitive_restart_enable, const GraphicsPrograms& programs) {
	const auto& vs_input_info  = vertex_info.front();
	const auto& vertex_program = programs.vertex[0];
	const auto& pixel_program  = programs.pixel;
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Gfx)", profiler::colors::DeepOrangeA200);

	EXIT_IF(colors.size() > RENDER_COLOR_ATTACHMENTS_MAX);
	EXIT_IF(!vertex_program);
	const bool ps_active = ps_input_info != nullptr;
	EXIT_IF(ps_active && !pixel_program);
	const auto color_count = static_cast<uint32_t>(colors.size());

	auto&             ctx = command.GetRegisters();

	const HW::ModeControl& mc = ctx.GetModeControl();

	const auto vs_id = vertex_program.id;
	const auto ps_id = ps_active ? pixel_program.id : 0;

	GraphicsPipelineKey key {};
	for (uint32_t i = 0; i < programs.vertex.size(); i++) {
		key.vertex_shader_ids[i] = programs.vertex[i].id;
	}
	key.ps_shader_id            = ps_id;
	auto& static_params         = key.static_params;
	auto& rendering             = key.rendering;
	rendering.color_count       = 0;
	uint32_t attachment_samples = 0;
	for (uint32_t i = 0; i < color_count; i++) {
		const auto slot = colors[i].target_slot;
		EXIT_IF(slot >= RENDER_COLOR_ATTACHMENTS_MAX);
		rendering.color_count = std::max(rendering.color_count, slot + 1);
		EXIT_IF(!colors[i].image_id || colors[i].desc.view_info.format == vk::Format::eUndefined);
		static_params.color_mask[slot] = colors[i].export_mapping.ApplyMask(
		    render_target_mask_slot(ctx.GetRenderTargetMask(), colors[i].target_slot));
		if (colors[i].desc.info.guest_format == Prospero::BufferFormat::k11_11_10UNorm) {
			static_params.color_mask[slot] = 1u;
		}
		rendering.color_formats[slot] = colors[i].desc.view_info.format;
		if (attachment_samples == 0) {
			attachment_samples = colors[i].desc.info.samples;
		} else if (attachment_samples != colors[i].desc.info.samples) {
			EXIT("mixed color attachment sample counts are unsupported: %u and %u\n",
			     attachment_samples, colors[i].desc.info.samples);
		}
		const auto& rt                        = ctx.GetRenderTarget(colors[i].target_slot);
		const auto& bc                        = ctx.GetBlendControl(colors[i].target_slot);
		auto alpha_source = ShaderAlphaBlendSource::None;
		if (slot == 0 && ps_input_info != nullptr) {
			alpha_source = ps_input_info->alpha_blend_source;
		}
		static_params.blend_enable[slot] = bc.enable && !rt.info.blend_bypass;
		if (static_params.blend_enable[slot] && alpha_source == ShaderAlphaBlendSource::None &&
		    ClassifyBlendMapping(bc, colors[i].export_mapping) != BlendMappingSupport::Direct) {
			static_params.blend_enable[slot] = false;
			static std::atomic_bool warned = false;
			if (!warned.exchange(true, std::memory_order_relaxed)) {
				Log::WriteToConsoleAndLog(fmt::format(
				    "Warning: blending disabled for unsupported color mapping "
				    "(slot={} mapping=0x{:02x} color={}/{} alpha={}/{} separate={}).\n",
				    slot, colors[i].export_mapping.packed, bc.color_srcblend, bc.color_destblend,
				    bc.alpha_srcblend, bc.alpha_destblend, bc.separate_alpha_blend ? 1 : 0));
			}
		}
		if (static_params.blend_enable[slot]) {
			auto blend = bc;
			switch (alpha_source) {
				case ShaderAlphaBlendSource::SourceAlpha:
					blend.color_srcblend  = RemapSourceAlphaFactor(blend.color_srcblend);
					blend.color_destblend = RemapSourceAlphaFactor(blend.color_destblend);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::SourceAlphaOne:
				case ShaderAlphaBlendSource::SourceAlphaZero:
					// The second source carries the mapped source factor; its alpha stays logical Sa.
					blend.color_srcblend = static_cast<uint8_t>(Prospero::BlendFactor::kSrc1Color);
					blend.color_destblend =
					    static_cast<uint8_t>(Prospero::BlendFactor::kOneMinusSrc1Alpha);
					blend.separate_alpha_blend = false;
					break;
				case ShaderAlphaBlendSource::None: break;
			}
			static_params.color_srcblend[slot]       = blend.color_srcblend;
			static_params.color_comb_fcn[slot]       = blend.color_comb_fcn;
			static_params.color_destblend[slot]      = blend.color_destblend;
			static_params.separate_alpha_blend[slot] = blend.separate_alpha_blend;
			if (blend.separate_alpha_blend) {
				static_params.alpha_srcblend[slot]  = blend.alpha_srcblend;
				static_params.alpha_comb_fcn[slot]  = blend.alpha_comb_fcn;
				static_params.alpha_destblend[slot] = blend.alpha_destblend;
			}
		}
	}
	const bool with_depth =
	    depth.desc.view_info.format != vk::Format::eUndefined && static_cast<bool>(depth.image_id);
	if (with_depth) {
		const auto aspects       = ImageViewOps::DepthAspectMask(depth.desc.view_info.format);
		rendering.depth_format   = aspects & vk::ImageAspectFlagBits::eDepth
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		rendering.stencil_format = aspects & vk::ImageAspectFlagBits::eStencil
		                               ? depth.desc.view_info.format
		                               : vk::Format::eUndefined;
		if (attachment_samples == 0) {
			attachment_samples = depth.desc.info.samples;
		} else if (attachment_samples != depth.desc.info.samples) {
			EXIT("mixed color/depth sample counts are unsupported: %u and %u\n", attachment_samples,
			     depth.desc.info.samples);
		}
	}
	if (color_count == 0 && !with_depth) {
		attachment_samples = render_sample_count(ctx.GetAaConfig().msaa_num_samples);
		EXIT_IF(!static_cast<bool>(
		    m_graphics.GetPhysicalDeviceProperties().limits.framebufferNoAttachmentsSampleCounts &
		    vulkan_sample_count(attachment_samples)));
	}
	EXIT_IF(attachment_samples == 0 ||
	        vulkan_sample_count(attachment_samples) == vk::SampleCountFlagBits {});

	if (ps_active && depth.depth_test_enable && ps_input_info->ps_execute_on_noop) {
		static std::atomic<uint32_t> log_count {0};
		if (log_count.fetch_add(1, std::memory_order_relaxed) < 16) {
			LOGF("Pipeline: temporary: accepting EXEC_ON_NOOP with depth test enabled\n");
		}
	}

	const auto& clip_control               = ctx.GetClipControl();
	static_params.negative_one_to_one      = !clip_control.dx_clip_space;
	static_params.depth_clip_enable        = clip_control.IsZClipEnabled();
	static_params.topology                 = topology;
	static_params.primitive_restart_enable = primitive_restart_enable;
	static_params.samples                  = attachment_samples;
	static_params.sample_shading_enable =
	    ps_active && attachment_samples > 1 && ps_input_info->ps_sample_shading;
	if (static_params.sample_shading_enable && !m_graphics.sample_rate_shading_enabled) {
		EXIT("Pipeline: sample-rate shading is required but unsupported by the host\n");
	}
	const bool rect_list = Prospero::IsRectList(command.GetUserConfig().GetPrimType());
	static_params.cull_back  = !rect_list && mc.cull_back;
	static_params.cull_front = !rect_list && mc.cull_front;
	static_params.face       = mc.face;
	static_params.provoking_vtx_last = mc.provoking_vtx_last;
	static_params.polygon_mode =
	    ResolvePolygonMode(mc, static_params.cull_front, static_params.cull_back);

	if (vs_input_info.stage.program->stage != ShaderType::Mesh) {
		EXIT_IF(vs_input_info.buffers_num < 0 ||
		        vs_input_info.buffers_num > ShaderVertexInputInfo::RES_MAX ||
		        vs_input_info.resources_num < 0 ||
		        vs_input_info.resources_num > ShaderVertexInputInfo::RES_MAX);
		key.vertex_input.binding_count   = static_cast<uint8_t>(vs_input_info.buffers_num);
		key.vertex_input.attribute_count = static_cast<uint8_t>(vs_input_info.resources_num);
		for (int binding = 0; binding < vs_input_info.buffers_num; binding++) {
			const auto& buffer = vs_input_info.buffers[binding];
			key.vertex_input.bindings[binding] = {.stride   = buffer.stride,
			                                      .instance = buffer.fetch_index != 0};
		}
		for (int attribute = 0; attribute < vs_input_info.resources_num; attribute++) {
			const auto binding = vs_input_info.resources_dst[attribute].buffer_index;
			EXIT_IF(binding < 0 || binding >= vs_input_info.buffers_num);
			key.vertex_input.attributes[attribute] = {
			    .offset = static_cast<uint32_t>(vs_input_info.resources[attribute].Base48() -
			                                    vs_input_info.buffers[binding].addr),
			    .binding = static_cast<uint8_t>(binding),
			};
		}
	}

	if (auto iter = m_graphics_pipelines.find(key); iter != m_graphics_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(vs_input_info);
		if (ps_active) {
			ShaderDbgDumpInputInfo(*ps_input_info);
		}
		LOGF("PipelineTrace: shader modules VS=%" PRIu64 " module=%p PS=%" PRIu64 " module=%p\n",
		     vs_id, static_cast<void*>(vertex_program.module), ps_id,
		     static_cast<void*>(pixel_program.module));
	}

	auto cached = std::make_unique<Pipeline>();
	LogPipelineTrace("CreatePipelineInternal begin", vs_id, ps_id);
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, rendering, key.vertex_input, vertex_info,
		                       ps_input_info, programs, static_params, m_driver_cache);
	}
	LogPipelineTrace("CreatePipelineInternal done", vs_id, ps_id);

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	{
		std::scoped_lock lock(g_manifest_mutex);
		if (!g_manifest_graphics.contains(key)) {
			ManifestGraphicsEntry entry;
			entry.key = key;
			entry.mesh = vs_input_info.stage && vs_input_info.stage.program->stage == ShaderType::Mesh;
			entry.tessellation = vertex_info.size() == 3;
			entry.rect_list = !entry.mesh && !entry.tessellation && static_params.topology == vk::PrimitiveTopology::ePatchList;
			entry.patch_control_points = entry.tessellation ? vs_input_info.tess.input_control_points : 3u;
			for (size_t i = 0; i < vertex_info.size(); i++) {
				const auto* perm = m_program_cache->FindPermutation(programs.vertex[i].id);
				if (perm != nullptr && !perm->spirv.empty()) {
					entry.stages.push_back({vertex_info[i].logical_stage, perm->program.wave_size, perm->spirv});
					for (auto mask = perm->program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
						const auto index = std::countr_zero(mask);
						const auto kind = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index);
						vk::DescriptorSetLayoutBinding b {};
						b.binding         = ShaderRecompiler::IR::NativeBinding(perm->program.stage, kind);
						b.descriptorType  = NativeDescriptorType(kind);
						b.descriptorCount = perm->program.bindings.descriptor_counts[index];
						b.stageFlags      = NativeShaderStage(vertex_info[i].logical_stage);
						entry.bindings.push_back(b);
					}
				}
			}
			if (ps_active && programs.pixel) {
				const auto* perm = m_program_cache->FindPermutation(programs.pixel.id);
				if (perm != nullptr && !perm->spirv.empty()) {
					entry.stages.push_back({ShaderType::Pixel, perm->program.wave_size, perm->spirv});
					for (auto mask = perm->program.bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
						const auto index = std::countr_zero(mask);
						const auto kind = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index);
						vk::DescriptorSetLayoutBinding b {};
						b.binding         = ShaderRecompiler::IR::NativeBinding(perm->program.stage, kind);
						b.descriptorType  = NativeDescriptorType(kind);
						b.descriptorCount = perm->program.bindings.descriptor_counts[index];
						b.stageFlags      = vk::ShaderStageFlagBits::eFragment;
						entry.bindings.push_back(b);
					}
				}
			}
			for (uint32_t binding = 0; binding < key.vertex_input.binding_count; binding++) {
				vk::VertexInputBindingDescription bind_desc {};
				bind_desc.binding   = binding;
				bind_desc.stride    = key.vertex_input.bindings[binding].stride;
				bind_desc.inputRate = key.vertex_input.bindings[binding].instance ? vk::VertexInputRate::eInstance : vk::VertexInputRate::eVertex;
				entry.input_bindings.push_back(bind_desc);
			}
			for (uint32_t attr = 0; attr < key.vertex_input.attribute_count; attr++) {
				uint32_t attr_size = 4;
				vk::Format format = vk::Format::eUndefined;
				const auto registers_num = vs_input_info.resources_dst[attr].registers_num;
				const auto compiled_components = vs_input_info.stage.program->info.vertex_fetch_components[attr];
				const auto used_components = compiled_components > 0 ? static_cast<int>(compiled_components) : registers_num;
				GetInputFormat(vs_input_info.resources[attr], format, attr_size, static_cast<uint32_t>(used_components));
				vk::VertexInputAttributeDescription attr_desc {};
				attr_desc.location = attr;
				attr_desc.binding  = key.vertex_input.attributes[attr].binding;
				attr_desc.format   = format;
				attr_desc.offset   = key.vertex_input.attributes[attr].offset;
				entry.input_attributes.push_back(attr_desc);
			}
			g_manifest_graphics.emplace(key, std::move(entry));
			m_manifest_dirty = true;
		}
	}

	auto [iter, inserted] = m_graphics_pipelines.emplace(std::move(key), std::move(cached));
	EXIT_IF(!inserted);
	CheckpointDriverCache();

	return *iter->second;
}

PipelineCache::Pipeline&
PipelineCache::GetComputePipeline(const ShaderComputeInputInfo& input_info,
                                  const ShaderProgram&          compute_program) {
	KYTY_PROFILER_BLOCK("PipelineCache::CreatePipeline(Compute)", profiler::colors::RedA100);

	EXIT_IF(!compute_program);

	if (auto iter = m_compute_pipelines.find(compute_program.id);
	    iter != m_compute_pipelines.end()) {
		return *iter->second;
	}

	if (graphics_debug_dump_enabled()) {
		ShaderDbgDumpInputInfo(input_info);
	}

	auto cached = std::make_unique<Pipeline>();
	{
		Common::DebugCounters::ScopedTimer timer(Common::DebugCounters::Counter::PipelinesCreated,
		                                         Common::DebugCounters::Counter::PipelineCreateNs,
		                                         Common::DebugCounters::Peak::PipelineCreateNs);
		CreatePipelineInternal(m_graphics, *cached, input_info, compute_program.module,
		                       m_driver_cache);
	}

	EXIT_NOT_IMPLEMENTED(cached->pipeline == nullptr);
	EXIT_NOT_IMPLEMENTED(cached->pipeline_layout == nullptr);

	const auto* perm = m_program_cache->FindPermutation(compute_program.id);
	if (perm != nullptr && !perm->spirv.empty()) {
		std::scoped_lock lock(g_manifest_mutex);
		if (!g_manifest_compute.contains(compute_program.id)) {
			ManifestComputeEntry entry;
			entry.shader_id = compute_program.id;
			entry.wave_size = input_info.stage.program ? input_info.stage.program->wave_size : 64u;
			entry.spirv     = perm->spirv;
			if (input_info.stage.program != nullptr) {
				for (auto mask = input_info.stage.program->bindings.descriptor_mask; mask != 0; mask &= mask - 1) {
					const auto index = std::countr_zero(mask);
					const auto kind = static_cast<ShaderRecompiler::IR::DescriptorBindingKind>(index);
					vk::DescriptorSetLayoutBinding b {};
					b.binding         = ShaderRecompiler::IR::NativeBinding(input_info.stage.program->stage, kind);
					b.descriptorType  = NativeDescriptorType(kind);
					b.descriptorCount = input_info.stage.program->bindings.descriptor_counts[index];
					b.stageFlags      = vk::ShaderStageFlagBits::eCompute;
					entry.bindings.push_back(b);
				}
			}
			g_manifest_compute.emplace(compute_program.id, std::move(entry));
			m_manifest_dirty = true;
		}
	}

	auto [iter, inserted] = m_compute_pipelines.emplace(compute_program.id, std::move(cached));
	EXIT_IF(!inserted);
	CheckpointDriverCache();

	return *iter->second;
}
} // namespace Libs::Graphics
