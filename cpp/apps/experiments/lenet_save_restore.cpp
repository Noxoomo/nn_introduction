#include <experiments/core/networks/lenet.h>
#include <experiments/core/model.h>

#include <torch/torch.h>

#include <memory>
#include <iostream>
#include <cassert>


int main(int argc, char* argv[]) {
    using namespace experiments;

    auto lenet = std::make_shared<LeNet>();
    auto lenet2 = std::make_shared<LeNet>();

    torch::Tensor fakeBatch = torch::randn({10, 3, 32, 32}, torch::kFloat32);

    auto resBeforeSave = lenet->forward(fakeBatch);
    auto resBeforeSave2 = lenet2->forward(fakeBatch);

    assert(!resBeforeSave.equal(resBeforeSave2));

    std::cout << "saving net" << std::endl;
    torch::save(lenet, "lenet_save_restore_test.pt");
    std::cout << "restoring net" << std::endl;
    torch::load(lenet2, "lenet_save_restore_test.pt");
    std::cout << "done" << std::endl;

    auto resAfterLoad = lenet2->forward(fakeBatch);

    std::cout << resBeforeSave << std::endl;
    std::cout << resAfterLoad << std::endl;


    assert(resBeforeSave.equal(resAfterLoad));
}
