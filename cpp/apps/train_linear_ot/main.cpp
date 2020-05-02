#include <memory>

#include <data/dataset.h>
#include <data/load_data.h>

#include <data/grid_builder.h>
#include <methods/boosting.h>
#include <methods/boosting_weak_target_factory.h>
#include <methods/greedy_linear_oblivious_trees.h>
#include <targets/cross_entropy.h>
#include <util/json.h>

inline std::unique_ptr<GreedyLinearObliviousTreeLearner> createWeakLearner(GridPtr grid, GreedyLinearObliviousTreeLearnerOptions opts) {
  return std::make_unique<GreedyLinearObliviousTreeLearner>(grid, opts);
}

inline std::unique_ptr<EmpiricalTargetFactory> createWeakTarget() {
  return std::make_unique<GradientBoostingWeakTargetFactory>();
}

inline std::unique_ptr<EmpiricalTargetFactory>
createBootstrapWeakTarget(BootstrapOptions options) {
  return std::make_unique<GradientBoostingBootstrappedWeakTargetFactory>(
      options);
}

int main(int /*argc*/, char * /*argv*/[]) {
  auto start = std::chrono::system_clock::now();
  auto params = readJson("train_config.json");
  torch::set_num_threads(params.value("num_threads", std::thread::hardware_concurrency()));
  //    auto ds =
  //    loadFeaturesTxt("/Users/noxoomo/Projects/moscow_learn_200k.tsv"); auto
  //    test = loadFeaturesTxt("/Users/noxoomo/Projects/moscow_test.tsv");
  auto ds = loadFeaturesTxt(params.value("train", "features.txt"));
  auto test = loadFeaturesTxt(params.value("test", "featuresTest.txt"));

  std::cout << " load data in memory " << std::endl;
  BinarizationConfig config = BinarizationConfig::fromJson(params["grid_config"]);
  auto grid = buildGrid(ds, config);
  std::cout << " build grid " << std::endl;

  BoostingConfig boostingConfig =
      BoostingConfig::fromJson(params["boosting_config"]);
  BootstrapOptions bootstrapOptions =
      BootstrapOptions::fromJson(params["bootstrap_options"]);
  Boosting boosting(
      boostingConfig, createBootstrapWeakTarget(bootstrapOptions),
      createWeakLearner(grid, GreedyLinearObliviousTreeLearnerOptions::fromJson(params["tree_config"])));

  std::string target = params.value("target", "mse");
  std::unique_ptr<Target> objective;
  auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
  if (target == "mse") {
    metricsCalcer->addMetric(L2(test), "l2_test");
    boosting.addListener(metricsCalcer);
    objective.reset(new L2(ds));
  } else if (target == "ce") {
    metricsCalcer->addMetric(CrossEntropy(test), "CE_test");
    objective.reset(new CrossEntropy(ds));
  } else {
    VERIFY(false, "Unknown target " << target);
  }
  boosting.addListener(metricsCalcer);

  auto ensemble = boosting.fit(ds, *objective);
  std::cout << "total time "
            << std::chrono::duration<double>(std::chrono::system_clock::now() -
                                             start)
                   .count()
            << std::endl;
}
