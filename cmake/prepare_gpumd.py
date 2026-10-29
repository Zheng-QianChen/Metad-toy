"""Copy GPUMD v5.9.1 sources and add bounded, optional EnhancedSampling hooks."""
import argparse
import hashlib
import re
import shutil
from pathlib import Path

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError("GPUMD v5.9.1 patch anchor must occur once: " + old[:100])
    return text.replace(old, new, 1)

def prepare(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if source == destination or source in destination.parents:
        raise ValueError("Use a separate build directory outside GPUMD sources")
    # 相同源码和补丁脚本复用构建副本，避免配置阶段触发上游头文件全量重编译。
    signature = hashlib.sha256(Path(__file__).read_bytes())
    for path in sorted((source / "src").rglob("*")):
        if path.is_file() and path.suffix in (".cu", ".cuh", ".h"):
            signature.update(str(path.relative_to(source)).encode())
            signature.update(path.read_bytes())
    stamp = destination / ".metadtoy-bridge-stamp"
    digest = signature.hexdigest()
    if stamp.exists() and stamp.read_text() == digest:
        return
    for component in ("main_gpumd", "minimize", "phonon", "integrate", "mc", "force", "measure", "model", "utilities"):
        shutil.copytree(source / "src" / component, destination / "src" / component,
                        dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.o", "*.obj"))
    shutil.copytree(source / "LICENSES", destination / "LICENSES", dirs_exist_ok=True)
    if (source / "UPSTREAM.json").exists():
        shutil.copyfile(source / "UPSTREAM.json", destination / "UPSTREAM.json")
    path = destination / "src/measure/enhanced_sampling.cuh"
    text = replace_once(path.read_text(), '#include "action.cuh"',
        '#include "action.cuh"\n#ifdef ZQC_USE_METADTOY\n#include "zqc_gpumd_bridge.cuh"\n#endif')
    text = replace_once(text, "  EnhancedSamplingEngine engine_;",
        "  EnhancedSamplingEngine engine_;\n#ifdef ZQC_USE_METADTOY\n"
        "  std::unique_ptr<MetaDToyGPUMDBridge> metadtoy_;\n#endif")
    path.write_text(text)
    path = destination / "src/measure/enhanced_sampling.cu"
    text = replace_once(path.read_text(), "  input_filename_ = tokens[1];",
        "  input_filename_ = tokens[1];\n#ifdef ZQC_USE_METADTOY\n"
        "  metadtoy_ = MetaDToyGPUMDBridge::create_if_requested(input_filename_);\n"
        "  if (metadtoy_) return;\n#endif")
    hooks = {
        "pre_run": "metadtoy_->pre_run(integrate, atom, box)",
        "setup_force": "metadtoy_->evaluate(0, false, atom, box)",
        "post_force": "metadtoy_->evaluate(static_cast<long long>(step) + 1, true, atom, box)",
        "end_of_step": "metadtoy_->end_of_step(static_cast<long long>(step) + 1)",
        "post_run": "metadtoy_->post_run()"}
    for name, call in hooks.items():
        pattern = r"(void EnhancedSamplingAction::" + name + r"\([^{}]*\)\n\{)"
        text, count = re.subn(pattern, lambda m: m[1] +
            "\n#ifdef ZQC_USE_METADTOY\n  if (metadtoy_) { " + call + "; return; }\n#endif", text)
        if count != 1:
            raise RuntimeError("Missing/ambiguous GPUMD callback: " + name)
    path.write_text(text)
    path = destination / "src/force/force.cu"
    text = replace_once(path.read_text(), '#include "force.cuh"', '#include "force.cuh"\n'
        '#ifdef ZQC_GPUMD_VALIDATION_MORSE\n#include "morse_validation.cuh"\n#endif')
    anchor = '  } else if (strcmp(potential_name, "lj") == 0) {'
    text = replace_once(text, anchor, '#ifdef ZQC_GPUMD_VALIDATION_MORSE\n'
        '  } else if (strcmp(potential_name, "metadtoy_morse") == 0) {\n'
        '    potential.reset(new MetaDToyMorseValidation(fid_potential, num_types, number_of_atoms));\n'
        '#endif\n' + anchor)
    path.write_text(text)
    stamp.write_text(digest)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    arguments = parser.parse_args()
    prepare(arguments.source, arguments.destination)
