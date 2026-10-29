#include "zqc_cv.h"
#include <stdexcept>

namespace MetaD_zqc {
std::map<std::string, CVFactory::CreatorFunc>& CVFactory::get_registry() {
    static std::map<std::string, CreatorFunc> registry;
    return registry;
}
void CVFactory::register_cv(std::string name, CreatorFunc creator) {
    get_registry()[std::move(name)] = creator;
}
CV* CVFactory::create(std::string name, CVContext& context,
                      int argument_count, char** arguments, int& argument_index) {
    const auto found = get_registry().find(name);
    if (found == get_registry().end())
        throw std::invalid_argument("Unknown CV type: " + name);
    return found->second(context, argument_count, arguments, argument_index);
}
} // namespace MetaD_zqc
