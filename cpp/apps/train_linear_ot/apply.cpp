#include <memory>

#include <data/dataset.h>
#include <data/load_data.h>

#include <methods/boosting.h>
#include <methods/boosting_weak_target_factory.h>
#include <methods/greedy_linear_oblivious_trees.h>
#include <util/json.h>
#include <data/grid_builder.h>

int main(int /*argc*/, char* argv[]) {
    auto params = readJson(argv[1]);
    torch::set_num_threads(params.value("num_threads", std::thread::hardware_concurrency()));

    auto ds = loadFeaturesTxt(params.value("dataset", "featuresTest.txt"));

    // This is required at the moment
    ds.addBiasColumn();

    if (params.value("normalize", false)) {
        Vec mu(ds.featuresCount());
        Vec sd(ds.featuresCount());
        ds.computeNormalization(mu.arrayRef(), sd.arrayRef());
        ds.normalizeColumns(mu.arrayRef(), sd.arrayRef());
    }

    GridPtr grid;

    if (params.contains("build_grid_from")) {
        auto gridds = loadFeaturesTxt(params["build_grid_from"]);
        auto binarizationCfg = BinarizationConfig::fromJson(params);
        // TODO should we also normalize it? I guess yes, but I only use it for tests now
        grid = buildGrid(gridds, binarizationCfg);
    }

    std::string checkpoint_path = params["checkpoint_from_file"];
    std::ifstream fin(checkpoint_path, std::ios::binary);
    std::shared_ptr<Ensemble> ensemble = Ensemble::deserialize(fin, [&]() {
        return LinearObliviousTree::deserialize(fin, grid);
    });
    fin.close();

    std::cout << "restored an ensemble of size " << ensemble->size() << std::endl;

    Mx prediction(ds.samplesCount(), 1);
    ensemble->apply(ds, prediction);

    L2 target(ds);
    std::cout << "L2: " << target.value(prediction) << std::endl;

    if (params.contains("save_predictions_to")) {
        std::string outPath = params["save_predictions_to"];
        std::ofstream out(outPath);
        auto predRef = prediction.arrayRef();
        for (int i = 0; i < (int)predRef.size(); ++i) {
            out << predRef[i] << "\n";
        }
        out.close();
    }
}
