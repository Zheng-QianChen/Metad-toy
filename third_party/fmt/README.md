# fmt 10.2.0

本目录保存 [fmt 上游 10.2.0](https://github.com/fmtlib/fmt/tree/10.2.0) 的 `include/fmt/*.h` 和原始 [MIT 许可证](LICENSE)。头文件保持上游内容。

来源归档：`https://codeload.github.com/fmtlib/fmt/zip/refs/tags/10.2.0`。
归档 SHA-256：`fa3e5d11027fe9c9b78c25dde8e560202445cf45d2317d4e4694997ca8143a77`。

`cmake/GraphDependencies.cmake` 创建 header-only `zqc_fmt` 目标，Graph 的 Host 源码生成器使用其包含路径。构建无需下载依赖或配置系统 fmt。NVRTC 输入只包含生成的设备函数及 kernel，无需 fmt 头文件。LAMMPS 适配层仍包含宿主自身头文件及其内部依赖。
