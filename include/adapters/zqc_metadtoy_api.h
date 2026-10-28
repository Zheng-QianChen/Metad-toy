#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ZQC_METADTOY_ABI_VERSION 1u

/* C ABI 的数组描述。data 借用宿主 CUDA 内存，stride/capacity 均以元素计。
 * double 数组用于坐标、力、virial 和势能；types 使用 int32_t。
 * 生命周期覆盖 pre_run/evaluate 调用及其提交的 GPU 消费操作；
 * evaluate 返回后，库已在 frame.stream 上插入回写完成依赖。 */
typedef struct ZqcArrayV1 {
    void* data;
    uint64_t items, components, item_stride, component_stride, capacity;
} ZqcArrayV1;

/* 固定顺序的单进程原子帧。virial 的九分量顺序为
 * xx yy zz xy xz yz yx zx zy。stream 是借用的 cudaStream_t（nullptr 为默认流）。
 * positions/forces/virials/potential 必填，types 可为空。
 * size 与 abi_version 共同校验布局；破坏布局的修改需要提升 ABI 版本。 */
typedef struct ZqcFrameV1 {
    uint32_t size, abi_version;
    ZqcArrayV1 positions, forces, virials, potential, types;
    double box_lengths[3];
    int32_t periodic[3];
    int64_t step;
    void* stream;
} ZqcFrameV1;

typedef struct ZqcRunV1 {
    uint32_t size;
    int32_t has_thermostat;
    double temperature;
} ZqcRunV1;

/* 回调在调用线程同步执行。message 仅在回调期间有效；宿主可复制或直接输出。
 * 回调应正常返回，C++ 异常须在宿主内部处理。 */
typedef void (*ZqcLogV1)(void* user, const char* message);
typedef struct ZqcHostV1 {
    uint32_t size;
    ZqcLogV1 log;
    void* user;
} ZqcHostV1;

/* 所有状态在库内申请、在 destroy 内释放；异常转换为非零返回码。
 * error 是宿主持有的 UTF-8 缓冲，capacity 包含结尾的 NUL。
 * 同一 handle 的调用需串行；销毁前需完成 post_run。
 * 函数表和 handle 的寿命覆盖当前进程，库升级在下一次启动时生效。 */
typedef struct ZqcApiV1 {
    uint32_t size, abi_version;
    int (*create)(const char*, const ZqcHostV1*, void**, char*, uint64_t);
    int (*pre_run)(void*, const ZqcFrameV1*, const ZqcRunV1*, char*, uint64_t);
    int (*evaluate)(void*, const ZqcFrameV1*, int32_t integrating, char*, uint64_t);
    int (*end_of_step)(void*, int64_t, char*, uint64_t);
    int (*post_run)(void*, char*, uint64_t);
    void (*destroy)(void*);
} ZqcApiV1;

/* 返回库静态持有的函数表；不支持请求版本时返回 NULL。 */
const ZqcApiV1* zqc_metadtoy_get_api(uint32_t requested_version);

#ifdef __cplusplus
}
#endif
