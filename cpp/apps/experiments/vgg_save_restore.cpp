#include <experiments/core/networks/vgg.h>
#include <experiments/core/model.h>

#include <torch/torch.h>

#include <memory>
#include <iostream>
#include <cassert>


int main(int argc, char* argv[]) {
    using namespace experiments;

    auto vgg = std::make_shared<Vgg>(VggConfiguration::Vgg16);
    auto vgg2 = std::make_shared<Vgg>(VggConfiguration::Vgg16);

    torch::Tensor fakeBatch = torch::randn({10, 3, 32, 32}, torch::kFloat32);

    auto resBeforeSave = vgg->forward(fakeBatch);
    auto resBeforeSave2 = vgg2->forward(fakeBatch);

    assert(!resBeforeSave.equal(resBeforeSave2));

    std::cout << "saving net" << std::endl;
    torch::save(vgg, "vgg_save_restore_test.pt");
    std::cout << "restoring net" << std::endl;
    torch::load(vgg2, "vgg_save_restore_test.pt");
    std::cout << "done" << std::endl;

    auto resAfterLoad = vgg2->forward(fakeBatch);

    std::cout << resBeforeSave << std::endl;
    std::cout << resAfterLoad << std::endl;

    assert(resBeforeSave.equal(resAfterLoad));
}
