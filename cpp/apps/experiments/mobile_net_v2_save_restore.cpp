#include <experiments/core/networks/mobile_net_v2.h>
#include <experiments/core/model.h>

#include <torch/torch.h>

#include <memory>
#include <iostream>
#include <cassert>

int main(int argc, char* argv[]) {
    using namespace experiments;

    auto mNet = std::make_shared<MobileNetV2>();
    auto mNet2 = std::make_shared<MobileNetV2>();

    torch::Tensor fakeBatch = torch::randn({10, 3, 32, 32}, torch::kFloat32);

    auto resBeforeSave = mNet->forward(fakeBatch);
    auto resBeforeSave2 = mNet2->forward(fakeBatch);

    assert(!resBeforeSave.equal(resBeforeSave2));

    std::cout << "saving net" << std::endl;
    torch::save(mNet, "mobile_net_v2_save_restore_test.pt");
    std::cout << "restoring net" << std::endl;
    torch::load(mNet2, "mobile_net_v2_save_restore_test.pt");
    std::cout << "done" << std::endl;

    auto resAfterLoad = mNet2->forward(fakeBatch);

    std::cout << resBeforeSave << std::endl;
    std::cout << resAfterLoad << std::endl;

    assert(resBeforeSave.equal(resAfterLoad));
}
