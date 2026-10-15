#include "zqc_graph.cuh"
#include "zqc_graph_device.cuh"

#include <cmath>
#include <limits>
#include <fmt/printf.h>

constexpr int block_size = 256;
static int grid_size(int n) { return (n + (block_size-1)) / block_size; }

static __global__ void k_mul(const double* a, const double* b, double* y, int n);
static __global__ void k_mul_bwd(const double* adj, const double* a, const double* b,
                          double* a_adj, double* b_adj, int n);
static __global__ void k_add(const double* a, const double* b, double* y, int n);
static __global__ void k_add_bwd(const double* adj, double* a_adj, double* b_adj, int n);
static __global__ void k_neg(const double* a, double* y, int n);
static __global__ void k_neg_bwd(const double* adj, double* a_adj, int n);
static __global__ void k_linear(const double* a, const double k0, const double b0, double* y, int n);
static __global__ void k_linear_bwd(const double* adj, const double* a,
                            const double k0, const double b0,
                          double* a_adj, int n);
static __global__ void k_exp(const double* a, double* y, int n);
static __global__ void k_exp_bwd(const double* adj, const double* a, double* a_adj, int n);
static __global__ void k_constant_one(const double* a, double* y, int n);
static __global__ void k_constant_one_bwd(const double* adj, const double* a, double* a_adj, int n);
static __global__ void k_step_switch(const double* a, double* y, double r0, int n);
static __global__ void k_step_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, int n);
static __global__ void k_fermi(const double* a, double* y, double r0, double alpha, int n);
static __global__ void k_fermi_bwd(const double* adj, const double* a, double* a_adj, double r0, double alpha, int n);
static __global__ void k_tanh_switch(const double* a, double* y, double r0, double alpha, int n);
static __global__ void k_tanh_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, double alpha, int n);
static __global__ void k_rational_switch(const double* a, double* y, double r0, int power_n, int power_m, double d0, int n);
static __global__ void k_rational_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, int power_n, int power_m, double d0, int n);

static __global__ void k_ave_bwd(double seed, int n, double* src_adj);


Node::Node(ValueDesc desc)
    : value_desc(desc){
    if (value_desc.components <= 0)
        throw std::runtime_error("Invalid component count");
}

bool Node::is_reduction() const { return false; }
const Node* Node::kernel_output() const { return this; }
bool AVE::is_reduction() const { return true; }
const Node* AVE::kernel_output() const { return src; }
void Node::bind_storage(double* device_value, double* device_adjoint) {
    val = value_desc.domain == Domain::Global ? &host_value : device_value;
    adj = value_desc.domain == Domain::Global ? &host_adjoint : device_adjoint;
}
void Node::reduce_host(const double*, int) {
    throw std::logic_error("Node: Host reduction is not implemented");
}
double Node::reduction_seed() const {
    throw std::logic_error("Node: reduction backward is not implemented");
}
static int device_count(const Node* input) {
    return input->value_desc.domain == Domain::PerAtom ? input->n : 0;
}

ValueDesc infer_broadcast(ValueDesc a, ValueDesc b){
    if (a.components <= 0 || b.components <= 0)
        throw std::runtime_error("Invalid component count");

    int components;

    if (a.components == b.components) {
        components = a.components;
    } else if (a.components == 1) {
        components = b.components;
    } else if (b.components == 1) {
        components = a.components;
    } else {
        throw std::runtime_error(
            "Incompatible component counts for broadcasting");
    }

    // 任意一个输入是逐原子值，输出就是逐原子值。
    Domain domain =
        (a.domain == Domain::PerAtom ||
         b.domain == Domain::PerAtom)
        ? Domain::PerAtom
        : Domain::Global;

    return {domain, components};
}

// 叶子节点(最终节点)
Leaf::Leaf(GpuBuffer<double>& values, int count,
                  ValueDesc desc)
        : Node(desc), abuf(desc.domain == Domain::PerAtom ? checked_count(count, desc) : 0),
          input_buffer_(&values){
    n = count;
    refresh_buffers();
}

Leaf::Leaf(GpuBuffer<double>& values, GpuBuffer<double>& gradients, int count,
         ValueDesc desc)
        : Node(desc), input_buffer_(&values), gradient_buffer_(&gradients){
    n = checked_count(count, desc);
    refresh_buffers();
}

Leaf::Leaf(const std::vector<double>& host,
                  ValueDesc desc)
        : Node(desc), xbuf(desc.domain == Domain::PerAtom ? checked_count(host.size(), desc) : 0),
          abuf(desc.domain == Domain::PerAtom ? host.size() : 0){
    n = checked_count(host.size(), desc);
    if (desc.domain == Domain::Global) {
        if (desc.components != 1) throw std::runtime_error("Leaf: only Global scalars are supported");
        host_value = host[0];
        bind_storage(nullptr, nullptr);
        return;
    }
    if (n) graph_check_cuda(cudaMemcpy(xbuf.ptr, host.data(), host.size() * sizeof(double),
                               cudaMemcpyHostToDevice));
    refresh_buffers();
}

void Leaf::refresh_buffers(){
    if (value_desc.domain == Domain::Global) {
        checked_count(n, value_desc);
        if (value_desc.components != 1) throw std::runtime_error("Leaf: only Global scalars are supported");
        if (input_buffer_) {
            if (!input_buffer_->ptr || input_buffer_->capacity < 1)
                throw std::runtime_error("Leaf: Global input buffer is too small");
            graph_check_cuda(cudaMemcpy(&host_value, input_buffer_->ptr, sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (gradient_buffer_ && (!gradient_buffer_->ptr || gradient_buffer_->capacity < 1))
            throw std::runtime_error("Leaf: Global gradient buffer is too small");
        bind_storage(nullptr, nullptr);
        return;
    }
    auto& values = input_buffer_ ? *input_buffer_ : xbuf;
    auto& gradients = gradient_buffer_ ? *gradient_buffer_ : abuf;
    if ((n && !values.ptr) || values.capacity < static_cast<size_t>(n))
        throw std::runtime_error("Leaf: input buffer is too small");
    if ((n && !gradients.ptr) || gradients.capacity < static_cast<size_t>(n))
        throw std::runtime_error("Leaf: gradient buffer is too small");
    val = values.ptr;
    adj = gradients.ptr;
    if (g) g = adj;
}

void Leaf::forward(){ refresh_buffers(); }

void Leaf::backward(){
    g = adj;
    if (value_desc.domain == Domain::Global && gradient_buffer_)
        graph_check_cuda(cudaMemcpy(gradient_buffer_->ptr, adj, sizeof(double), cudaMemcpyHostToDevice));
}

int Leaf::checked_count(size_t count, ValueDesc desc){
    if (count > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::runtime_error("Leaf: invalid element count");
    if (desc.components <= 0)
        throw std::runtime_error("Leaf: invalid components");
    if (desc.domain == Domain::Global) {
        if (count != static_cast<size_t>(desc.components))
            throw std::runtime_error("Leaf: global size mismatch");
    } else if (count % static_cast<size_t>(desc.components) != 0) {
        throw std::runtime_error("Leaf: incomplete atom components");
    }
    return static_cast<int>(count);
}
std::string Leaf::emit_expr(Codegen& cg) const{
    auto it = cg.value_names.find(this);
    if (it != cg.value_names.end())
        return it->second;

    std::string name = fmt::sprintf(
        "v%d", cg.next_val_id++);

    cg.value_names.emplace(this, name);
    cg.inputs.emplace_back(this, name);

    return name;
}
std::string Leaf::emit_expr_bwd(Codegen&) const{
    return {};
}
std::vector<const Node*> Leaf::inputs() const {
    return {};
}

// 逐原子乘的计算节点
Mul::Mul(Node* a_, Node* b_) : Node(infer_broadcast(a_->value_desc, b_->value_desc)),
        a(a_), b(b_), vbuf(device_count(a_)), abuf(device_count(a_)){
    if (a->value_desc.domain != b->value_desc.domain)
        throw std::runtime_error("Mul: implicit Global/PerAtom broadcasting is not supported");
    if (a->n != b->n) throw std::runtime_error("Mul: length mismatch");
    n = a->n;
    bind_storage(vbuf.ptr, abuf.ptr);
}

void Mul::forward(){
    if (value_desc.domain == Domain::Global) { *val = zqc_graph_mul(*a->val, *b->val); return; }
    if (n == 0) return;
    k_mul<<<grid_size(n), block_size>>>(a->val, b->val, val, n);
    sync_kernel();
}

void Mul::backward(){
    if (value_desc.domain == Domain::Global) { zqc_graph_mul_bwd(*adj, *a->val, *b->val, a->adj, b->adj); return; }
    if (n == 0) return;
    k_mul_bwd<<<grid_size(n), block_size>>>(adj, a->val, b->val, a->adj, b->adj, n);
    sync_kernel();
}
static __global__ void k_mul(const double* a, const double* b, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_mul(a[i], b[i]);
}
static __global__ void k_mul_bwd(const double* adj, const double* a, const double* b,
                          double* a_adj, double* b_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        zqc_graph_mul_bwd(adj[i], a[i], b[i], &a_adj[i], &b_adj[i]);
    }
}
std::string Mul::emit_expr(Codegen& cg) const{
    // 获取对应的变量名字符串
    // 其中，a和b是输入节点，可能是叶子节点，也可能是计算节点
    auto av = cg.value(a);
    auto bv = cg.value(b);
    return fmt::sprintf("zqc_graph_mul( %s , %s )", av, bv);
}
std::string Mul::emit_expr_bwd(Codegen& cg) const{
    auto av = cg.value(a);
    auto bv = cg.value(b);

    auto gv  = cg.adjoint(this);
    auto gav = cg.adjoint(a);
    auto gbv = cg.adjoint(b);
    return fmt::sprintf(
        "zqc_graph_mul_bwd(%s, %s, %s, &(%s), &(%s));\n",
        gv, av, bv, gav, gbv);
}
std::vector<const Node*> Mul::inputs() const {
    return {a, b};
}

// 逐原子加的计算节点
Add::Add(Node* a_, Node* b_) : Node(infer_broadcast(a_->value_desc, b_->value_desc)),
        a(a_), b(b_), vbuf(device_count(a_)), abuf(device_count(a_)){
    if (a->value_desc.domain != b->value_desc.domain)
        throw std::runtime_error("Add: implicit Global/PerAtom broadcasting is not supported");
    if (a->n != b->n) throw std::runtime_error("Add: length mismatch");
    n = a->n;
    bind_storage(vbuf.ptr, abuf.ptr);
}

void Add::forward(){
    if (value_desc.domain == Domain::Global) { *val = zqc_graph_add(*a->val, *b->val); return; }
    if (n == 0) return;
    k_add<<<grid_size(n), block_size>>>(a->val, b->val, val, n);
    sync_kernel();
}

void Add::backward(){
    if (value_desc.domain == Domain::Global) { zqc_graph_add_bwd(*adj, a->adj, b->adj); return; }
    if (n == 0) return;
    k_add_bwd<<<grid_size(n), block_size>>>(adj, a->adj, b->adj, n);
    sync_kernel();
}
static __global__ void k_add(const double* a, const double* b, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_add(a[i] , b[i]);
}
static __global__ void k_add_bwd(const double* adj, double* a_adj, double* b_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        zqc_graph_add_bwd(adj[i], &a_adj[i], &b_adj[i]);
    }
}
std::string Add::emit_expr(Codegen& cg) const{
    // 获取对应的变量名字符串
    // 其中，a和b是输入节点，可能是叶子节点，也可能是计算节点
    auto av = cg.value(a);
    auto bv = cg.value(b);
    return fmt::sprintf("zqc_graph_add( %s , %s )", av, bv);
}
std::string Add::emit_expr_bwd(Codegen& cg) const{
    auto gv  = cg.adjoint(this);
    auto gav = cg.adjoint(a);
    auto gbv = cg.adjoint(b);
    return fmt::sprintf(
        "zqc_graph_add_bwd(%s, &(%s), &(%s));\n",
        gv, gav, gbv);
}
std::vector<const Node*> Add::inputs() const {
    return {a, b};
}

// 逐元素取负：y = -a，保持输入的 domain、components 和长度。
Neg::Neg(Node* a_) : Node(a_->value_desc),
                            a(a_), vbuf(device_count(a_)), abuf(device_count(a_)){
    n = a->n;
    bind_storage(vbuf.ptr, abuf.ptr);
}

void Neg::forward(){
    if (value_desc.domain == Domain::Global) { *val = zqc_graph_neg(*a->val); return; }
    if (n == 0) return;
    k_neg<<<grid_size(n), block_size>>>(a->val, val, n);
    sync_kernel();
}

void Neg::backward(){
    if (value_desc.domain == Domain::Global) { zqc_graph_neg_bwd(*adj, a->adj); return; }
    if (n == 0) return;
    k_neg_bwd<<<grid_size(n), block_size>>>(adj, a->adj, n);
    sync_kernel();
}
static __global__ void k_neg(const double* a, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_neg(a[i]);
}
static __global__ void k_neg_bwd(const double* adj, double* a_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_neg_bwd(adj[i], &a_adj[i]);
}
std::string Neg::emit_expr(Codegen& cg) const {
    auto av = cg.value(a);
    return fmt::sprintf("zqc_graph_neg( %s )", av);
}
std::string Neg::emit_expr_bwd(Codegen& cg) const {
    auto gv = cg.adjoint(this);
    auto gav = cg.adjoint(a);
    return fmt::sprintf("zqc_graph_neg_bwd(%s, &(%s));\n", gv, gav);
}
std::vector<const Node*> Neg::inputs() const {
    return {a};
}

// 逐原子线性计算的计算节点
Linear::Linear(Node* a_, double k0_, double b0_) : Node(a_->value_desc),
                                            a(a_), vbuf(device_count(a_)), abuf(device_count(a_)),
                                            k0(k0_), b0(b0_){
    n = a->n;
    bind_storage(vbuf.ptr, abuf.ptr);
}

void Linear::forward(){
    if (value_desc.domain == Domain::Global) { *val = zqc_graph_linear(*a->val, k0, b0); return; }
    if (n == 0) return;
    k_linear<<<grid_size(n), block_size>>>(a->val, k0, b0, val, n);
    sync_kernel();
}

void Linear::backward(){
    if (value_desc.domain == Domain::Global) { zqc_graph_linear_bwd(*adj, k0, a->adj); return; }
    if (n == 0) return;
    k_linear_bwd<<<grid_size(n), block_size>>>(adj, a->val, k0, b0, a->adj, n);
    sync_kernel();
}
static __global__ void k_linear(const double* a, const double k0, const double b0, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_linear(a[i], k0, b0);
}
static __global__ void k_linear_bwd(const double* adj, const double* a,
                            const double k0, const double b0,
                          double* a_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        zqc_graph_linear_bwd(adj[i], k0, &a_adj[i]);
    }
}
std::string Linear::emit_expr(Codegen& cg) const{
    // 获取对应的变量名字符串
    // 其中，a和b是输入节点，可能是叶子节点，也可能是计算节点
    auto av = cg.value(a);
    return fmt::sprintf("zqc_graph_linear( %s , %.18g, %.18g )", av, k0, b0);
}
std::string Linear::emit_expr_bwd(Codegen& cg) const{
    auto av = cg.value(a);

    auto gv  = cg.adjoint(this);
    auto gav = cg.adjoint(a);
    return fmt::sprintf(
        "zqc_graph_linear_bwd(%s, %.18g, &(%s));\n",
        gv, k0, gav);
}
std::vector<const Node*> Linear::inputs() const {
    return {a};
}

// 逐元素指数函数
Exp::Exp(Node* a_) :  Node(a_->value_desc),
                    a(a_), vbuf(device_count(a_)), abuf(device_count(a_)){
    n = a->n;
    bind_storage(vbuf.ptr, abuf.ptr);
}

void Exp::forward(){
    if (value_desc.domain == Domain::Global) { *val = zqc_graph_exp(*a->val); return; }
    if (n == 0) return;
    k_exp<<<grid_size(n), block_size>>>(a->val, val, n);
    sync_kernel();
}

void Exp::backward(){
    if (value_desc.domain == Domain::Global) { zqc_graph_exp_bwd(*adj, *a->val, a->adj); return; }
    if (n == 0) return;
    k_exp_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, n);
    sync_kernel();
}
static __global__ void k_exp(const double* a, double* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_exp(a[i]);
}
static __global__ void k_exp_bwd(const double* adj, const double* a,
                          double* a_adj, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        zqc_graph_exp_bwd(adj[i], a[i], &a_adj[i]);
    }
}
std::string Exp::emit_expr(Codegen& cg) const{
    // 获取对应的变量名字符串
    // 其中，a和b是输入节点，可能是叶子节点，也可能是计算节点
    auto av = cg.value(a);
    return fmt::sprintf("zqc_graph_exp( %s )", av);
}
std::string Exp::emit_expr_bwd(Codegen& cg) const{
    auto av = cg.value(a);

    auto gv  = cg.adjoint(this);
    auto gav = cg.adjoint(a);
    return fmt::sprintf(
        "zqc_graph_exp_bwd(%s, %s, &(%s));\n",
        gv, av, gav);
}
std::vector<const Node*> Exp::inputs() const {
    return {a};
}

// Concrete elementwise nodes. Each captures only its own parameters.
ConstantOne::ConstantOne(Node* a_)
        : Node(a_->value_desc), a(a_), vbuf(device_count(a_)), abuf(device_count(a_)){
    n = a->n;
    bind_storage(vbuf.ptr, abuf.ptr);
}

void ConstantOne::forward(){
    if (value_desc.domain == Domain::Global) { *val = zqc_graph_constant_one(*a->val); return; }
    if (n == 0) return;
    k_constant_one<<<grid_size(n), block_size>>>(a->val, val, n);
    sync_kernel();
}

void ConstantOne::backward(){
    if (value_desc.domain == Domain::Global) { zqc_graph_constant_one_bwd(*adj, *a->val, a->adj); return; }
    if (n == 0) return;
    k_constant_one_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, n);
    sync_kernel();
}

std::string ConstantOne::emit_expr(Codegen& cg) const{
    return fmt::sprintf("zqc_graph_constant_one(%s)", cg.value(a));
}

std::string ConstantOne::emit_expr_bwd(Codegen& cg) const{
    const auto av = cg.value(a);
    const auto gv = cg.adjoint(this);
    const auto gav = cg.adjoint(a);
    return fmt::sprintf("zqc_graph_constant_one_bwd(%s, %s, &(%s));\n",
                        gv, av, gav);
}
static __global__ void k_constant_one(const double* a, double* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_constant_one(a[i]);
}
static __global__ void k_constant_one_bwd(const double* adj, const double* a, double* a_adj, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_constant_one_bwd(adj[i], a[i], &a_adj[i]);
}

StepSwitch::StepSwitch(Node* a_, double r0_)
        : Node(a_->value_desc), a(a_), r0(r0_), vbuf(device_count(a_)), abuf(device_count(a_)){
    if (!std::isfinite(r0)) throw std::runtime_error("StepSwitch: r0 must be finite");
    n = a->n;
    bind_storage(vbuf.ptr, abuf.ptr);
}

void StepSwitch::forward(){
    if (value_desc.domain == Domain::Global) { *val = zqc_graph_step_switch(*a->val, r0); return; }
    if (n == 0) return;
    k_step_switch<<<grid_size(n), block_size>>>(a->val, val, r0, n);
    sync_kernel();
}

void StepSwitch::backward(){
    if (value_desc.domain == Domain::Global) { zqc_graph_step_switch_bwd(*adj, *a->val, r0, a->adj); return; }
    if (n == 0) return;
    k_step_switch_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, r0, n);
    sync_kernel();
}

std::string StepSwitch::emit_expr(Codegen& cg) const{
    return fmt::sprintf("zqc_graph_step_switch(%s, %.18g)", cg.value(a), r0);
}

std::string StepSwitch::emit_expr_bwd(Codegen& cg) const{
    const auto av = cg.value(a);
    const auto gv = cg.adjoint(this);
    const auto gav = cg.adjoint(a);
    return fmt::sprintf("zqc_graph_step_switch_bwd(%s, %s, %.18g, &(%s));\n",
                        gv, av, r0, gav);
}
static __global__ void k_step_switch(const double* a, double* y, double r0, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_step_switch(a[i], r0);
}
static __global__ void k_step_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_step_switch_bwd(adj[i], a[i], r0, &a_adj[i]);
}

Fermi::Fermi(Node* a_, double r0_, double alpha_)
        : Node(a_->value_desc), a(a_), r0(r0_), alpha(alpha_), vbuf(device_count(a_)), abuf(device_count(a_)){
    if (!std::isfinite(r0) || !std::isfinite(alpha)) throw std::runtime_error("Fermi: parameters must be finite");
    n = a->n;
    bind_storage(vbuf.ptr, abuf.ptr);
}

void Fermi::forward(){
    if (value_desc.domain == Domain::Global) { *val = zqc_graph_fermi(*a->val, r0, alpha); return; }
    if (n == 0) return;
    k_fermi<<<grid_size(n), block_size>>>(a->val, val, r0, alpha, n);
    sync_kernel();
}

void Fermi::backward(){
    if (value_desc.domain == Domain::Global) { zqc_graph_fermi_bwd(*adj, *a->val, r0, alpha, a->adj); return; }
    if (n == 0) return;
    k_fermi_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, r0, alpha, n);
    sync_kernel();
}

std::string Fermi::emit_expr(Codegen& cg) const{
    return fmt::sprintf("zqc_graph_fermi(%s, %.18g, %.18g)", cg.value(a), r0, alpha);
}

std::string Fermi::emit_expr_bwd(Codegen& cg) const{
    const auto av = cg.value(a);
    const auto gv = cg.adjoint(this);
    const auto gav = cg.adjoint(a);
    return fmt::sprintf("zqc_graph_fermi_bwd(%s, %s, %.18g, %.18g, &(%s));\n",
                        gv, av, r0, alpha, gav);
}
static __global__ void k_fermi(const double* a, double* y, double r0, double alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_fermi(a[i], r0, alpha);
}
static __global__ void k_fermi_bwd(const double* adj, const double* a, double* a_adj, double r0, double alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_fermi_bwd(adj[i], a[i], r0, alpha, &a_adj[i]);
}

TanhSwitch::TanhSwitch(Node* a_, double r0_, double alpha_)
        : Node(a_->value_desc), a(a_), r0(r0_), alpha(alpha_), vbuf(device_count(a_)), abuf(device_count(a_)){
    if (!std::isfinite(r0) || !std::isfinite(alpha)) throw std::runtime_error("TanhSwitch: parameters must be finite");
    n = a->n;
    bind_storage(vbuf.ptr, abuf.ptr);
}

void TanhSwitch::forward(){
    if (value_desc.domain == Domain::Global) { *val = zqc_graph_tanh_switch(*a->val, r0, alpha); return; }
    if (n == 0) return;
    k_tanh_switch<<<grid_size(n), block_size>>>(a->val, val, r0, alpha, n);
    sync_kernel();
}

void TanhSwitch::backward(){
    if (value_desc.domain == Domain::Global) { zqc_graph_tanh_switch_bwd(*adj, *a->val, r0, alpha, a->adj); return; }
    if (n == 0) return;
    k_tanh_switch_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, r0, alpha, n);
    sync_kernel();
}

std::string TanhSwitch::emit_expr(Codegen& cg) const{
    return fmt::sprintf("zqc_graph_tanh_switch(%s, %.18g, %.18g)", cg.value(a), r0, alpha);
}

std::string TanhSwitch::emit_expr_bwd(Codegen& cg) const{
    const auto av = cg.value(a);
    const auto gv = cg.adjoint(this);
    const auto gav = cg.adjoint(a);
    return fmt::sprintf("zqc_graph_tanh_switch_bwd(%s, %s, %.18g, %.18g, &(%s));\n",
                        gv, av, r0, alpha, gav);
}
static __global__ void k_tanh_switch(const double* a, double* y, double r0, double alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_tanh_switch(a[i], r0, alpha);
}
static __global__ void k_tanh_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, double alpha, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_tanh_switch_bwd(adj[i], a[i], r0, alpha, &a_adj[i]);
}

RationalSwitch::RationalSwitch(Node* a_, double r0_, int power_n_, int power_m_, double d0_)
        : Node(a_->value_desc), a(a_), r0(r0_), power_n(power_n_), power_m(power_m_), d0(d0_), vbuf(device_count(a_)), abuf(device_count(a_)){
    if (!std::isfinite(r0) || !std::isfinite(d0) || r0 <= 0.0 || power_n <= 0 || power_m <= 0)
        throw std::runtime_error("RationalSwitch: finite r0 > 0, finite d0, and positive powers required");
    n = a->n;
    bind_storage(vbuf.ptr, abuf.ptr);
}

void RationalSwitch::forward(){
    if (value_desc.domain == Domain::Global) { *val = zqc_graph_rational_switch(*a->val, r0, power_n, power_m, d0); return; }
    if (n == 0) return;
    k_rational_switch<<<grid_size(n), block_size>>>(a->val, val, r0, power_n, power_m, d0, n);
    sync_kernel();
}

void RationalSwitch::backward(){
    if (value_desc.domain == Domain::Global) { zqc_graph_rational_switch_bwd(*adj, *a->val, r0, power_n, power_m, d0, a->adj); return; }
    if (n == 0) return;
    k_rational_switch_bwd<<<grid_size(n), block_size>>>(adj, a->val, a->adj, r0, power_n, power_m, d0, n);
    sync_kernel();
}

std::string RationalSwitch::emit_expr(Codegen& cg) const{
    return fmt::sprintf("zqc_graph_rational_switch(%s, %.18g, %d, %d, %.18g)", cg.value(a), r0, power_n, power_m, d0);
}

std::string RationalSwitch::emit_expr_bwd(Codegen& cg) const{
    const auto av = cg.value(a);
    const auto gv = cg.adjoint(this);
    const auto gav = cg.adjoint(a);
    return fmt::sprintf("zqc_graph_rational_switch_bwd(%s, %s, %.18g, %d, %d, %.18g, &(%s));\n",
                        gv, av, r0, power_n, power_m, d0, gav);
}
static __global__ void k_rational_switch(const double* a, double* y, double r0, int power_n, int power_m, double d0, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = zqc_graph_rational_switch(a[i], r0, power_n, power_m, d0);
}
static __global__ void k_rational_switch_bwd(const double* adj, const double* a, double* a_adj, double r0, int power_n, int power_m, double d0, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) zqc_graph_rational_switch_bwd(adj[i], a[i], r0, power_n, power_m, d0, &a_adj[i]);
}

// Host 规约是执行边界：设备数据就绪后才进入 MPI collective。
AVE::AVE(Node* src_, MPI_Comm comm_)
    : Node({Domain::Global, 1}), src(src_), comm(comm_) {
    if (!src || src->value_desc.domain != Domain::PerAtom || src->value_desc.components != 1)
        throw std::runtime_error("AVE: expected PerAtom scalar input");
    n = 1;
    bind_storage(nullptr, nullptr);
}

void AVE::reduce_host(const double* values, int count) {
    global_count = 0;
    if (count < 0 || (count && !values)) throw std::runtime_error("AVE: invalid local data");
    double local_sum = 0.0, correction = 0.0;
    for (int i = 0; i < count; ++i) {
        const double adjusted = values[i] - correction;
        const double next = local_sum + adjusted;
        correction = (next - local_sum) - adjusted;
        local_sum = next;
    }
    reduce_sum_count(local_sum, count);
}

void AVE::reduce_sum_count(double local_sum, int local_count) {
    global_count = 0;
    if (local_count < 0) throw std::runtime_error("AVE: negative local count");
    // 两个量打成一个数组，一次 MPI_SUM 同时计算全局 sum 和 count。
    // count 在 double 的精确整数范围内传输；空 rank 以 {0, 0} 正常参与。
    const double local[2] = {local_sum, static_cast<double>(local_count)};
    double global[2] = {local[0], local[1]};
    if (comm != MPI_COMM_NULL) {
        int initialized = 0, finalized = 0;
        MPI_Initialized(&initialized);
        if (initialized) MPI_Finalized(&finalized);
        if (!initialized || finalized) throw std::runtime_error("AVE: MPI must be active");
        if (MPI_Allreduce(local, global, 2, MPI_DOUBLE, MPI_SUM, comm) != MPI_SUCCESS)
            throw std::runtime_error("AVE: MPI_Allreduce failed");
    }
    // 在 2^53 边界即拒绝，避免更大的整数舍入成 2^53 后被误当成精确计数。
    if (global[1] >= 9007199254740992.0)
        throw std::overflow_error("AVE: global count must be below 2^53 for packed reduction");
    if (global[1] == 0.0) throw std::runtime_error("AVE: the global input is empty");
    global_count = static_cast<long long>(global[1]);
    *val = global[0] / global[1];
}

double AVE::reduction_seed() const {
    if (global_count <= 0) throw std::logic_error("AVE: forward is required before backward");
    return *adj / static_cast<double>(global_count);
}

void AVE::forward() {
    std::vector<double> values(src->n);
    if (src->n) graph_check_cuda(cudaMemcpy(values.data(), src->val, src->n*sizeof(double), cudaMemcpyDeviceToHost));
    reduce_host(values.data(), src->n);
}
void AVE::backward() {
    const double seed = reduction_seed();
    if (src->n) {
        k_ave_bwd<<<grid_size(src->n), block_size>>>(seed, src->n, src->adj);
        sync_kernel();
    }
}
static __global__ void k_ave_bwd(double seed, int n, double* src_adj) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) src_adj[i] += seed;
}
std::string AVE::emit_expr(Codegen&) const { throw std::logic_error("AVE executes on Host"); }
std::string AVE::emit_expr_bwd(Codegen&) const { throw std::logic_error("AVE backward executes on Host"); }
std::vector<const Node*> AVE::inputs() const { return {src}; }

std::vector<const Node*> ConstantOne::inputs() const {
    return {a};
}

std::vector<const Node*> StepSwitch::inputs() const {
    return {a};
}

std::vector<const Node*> Fermi::inputs() const {
    return {a};
}

std::vector<const Node*> TanhSwitch::inputs() const {
    return {a};
}

std::vector<const Node*> RationalSwitch::inputs() const {
    return {a};
}

// Each node interprets its parsed arguments; punctuation belongs to the parser.

Node* Mul::create(Graph& graph, const GraphExpr& call) {
    call.require_args(2);
    Node* a = graph.create_node(call.args[0]);
    Node* b = graph.create_node(call.args[1]);
    return graph.make_node<Mul>(a, b);
}

Node* Add::create(Graph& graph, const GraphExpr& call) {
    call.require_args(2);
    Node* a = graph.create_node(call.args[0]);
    Node* b = graph.create_node(call.args[1]);
    return graph.make_node<Add>(a, b);
}

Node* Neg::create(Graph& graph, const GraphExpr& call) {
    call.require_args(1);
    Node* input = graph.create_node(call.args[0]);
    return graph.make_node<Neg>(input);
}

Node* Exp::create(Graph& graph, const GraphExpr& call) {
    call.require_args(1);
    Node* input = graph.create_node(call.args[0]);
    return graph.make_node<Exp>(input);
}

Node* ConstantOne::create(Graph& graph, const GraphExpr& call) {
    call.require_args(1);
    Node* input = graph.create_node(call.args[0]);
    return graph.make_node<ConstantOne>(input);
}

Node* AVE::create(Graph& graph, const GraphExpr& call) {
    call.require_args(1);
    Node* input = graph.create_node(call.args[0]);
    return graph.make_node<AVE>(input, graph.comm);
}

Node* Linear::create(Graph& graph, const GraphExpr& call) {
    call.require_args(3);
    const double k0 = call.number_arg(1);
    const double b0 = call.number_arg(2);
    Node* input = graph.create_node(call.args[0]);
    return graph.make_node<Linear>(input, k0, b0);
}

Node* StepSwitch::create(Graph& graph, const GraphExpr& call) {
    call.require_args(2);
    const double r0 = call.number_arg(1);
    Node* input = graph.create_node(call.args[0]);
    return graph.make_node<StepSwitch>(input, r0);
}

Node* Fermi::create(Graph& graph, const GraphExpr& call) {
    call.require_args(3);
    const double r0 = call.number_arg(1);
    const double alpha = call.number_arg(2);
    Node* input = graph.create_node(call.args[0]);
    return graph.make_node<Fermi>(input, r0, alpha);
}

Node* TanhSwitch::create(Graph& graph, const GraphExpr& call) {
    call.require_args(3);
    const double r0 = call.number_arg(1);
    const double alpha = call.number_arg(2);
    Node* input = graph.create_node(call.args[0]);
    return graph.make_node<TanhSwitch>(input, r0, alpha);
}

Node* RationalSwitch::create(Graph& graph, const GraphExpr& call) {
    if (call.args.size() != 4 && call.args.size() != 5)
        call.fail("rational expects (input, r0, n, m[, d0])");
    const double r0 = call.number_arg(1);
    const int power_n = call.integer_arg(2);
    const int power_m = call.integer_arg(3);
    const double d0 = call.args.size() == 5 ? call.number_arg(4) : 0.0;
    Node* input = graph.create_node(call.args[0]);
    return graph.make_node<RationalSwitch>(input, r0, power_n, power_m, d0);
}

Node* Graph::make_switch_node(Node* input, const MetaD_zqc::SwitchFunctionRequest& p) {
    if (!std::isfinite(p.r_0) || !std::isfinite(p.d_0) || !std::isfinite(p.alpha))
        throw std::runtime_error("Switch configuration: parameters must be finite");
    switch (p.type) {
        case MetaD_zqc::LINE: return make_node<ConstantOne>(input);
        case MetaD_zqc::STEP: return make_node<StepSwitch>(input, p.r_0);
        case MetaD_zqc::FERMI: return make_node<Fermi>(input, p.r_0, p.alpha);
        case MetaD_zqc::TANH_TYPE: return make_node<TanhSwitch>(input, p.r_0, p.alpha);
        case MetaD_zqc::RATIONAL: return make_node<RationalSwitch>(input, p.r_0, p.n, p.m, p.d_0);
        default: throw std::runtime_error("Switch configuration: invalid type");
    }
}

namespace {
Node* create_subtract(Graph& graph, const GraphExpr& call) {
    call.require_args(2);
    Node* a = graph.create_node(call.args[0]);
    Node* b = graph.create_node(call.args[1]);
    Node* negative = graph.make_node<Neg>(b);
    return graph.make_node<Add>(a, negative);
}

Node* create_switch(Graph& graph, const GraphExpr& call) {
    call.require_args(2);
    const std::string& name = call.symbol_arg(1);
    auto it = graph.switches.find(name);
    if (it == graph.switches.end()) call.fail("unknown switching function: " + name);
    Node* input = graph.create_node(call.args[0]);
    return graph.make_switch_node(input, it->second);
}
}

void register_builtin_graph_nodes(Graph& graph) {
    graph.register_node("+", Add::create);
    graph.register_node("-", create_subtract);
    graph.register_node("*", Mul::create);
    graph.register_node("neg", Neg::create);
    graph.register_node("exp", Exp::create);
    graph.register_node("linear", Linear::create);
    graph.register_node("constant_one", ConstantOne::create);
    graph.register_node("step", StepSwitch::create);
    graph.register_node("fermi", Fermi::create);
    graph.register_node("tanh", TanhSwitch::create);
    graph.register_node("rational", RationalSwitch::create);
    graph.register_node("sw_func", create_switch);
    graph.register_node(".MEAN", AVE::create);
    graph.register_node(".AVE", AVE::create);
    graph.register_node("AVE", AVE::create);
}
