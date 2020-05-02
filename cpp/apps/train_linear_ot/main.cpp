#include <memory>

#include <data/dataset.h>
#include <data/load_data.h>

#include <data/grid_builder.h>
#include <methods/boosting.h>
#include <methods/boosting_weak_target_factory.h>
#include <methods/greedy_linear_oblivious_trees.h>
#include <targets/cross_entropy.h>
#include <util/json.h>
#include <experiments/core/linear_trees_booster.h>

inline std::unique_ptr<GreedyLinearObliviousTreeLearner> createWeakLearner(GridPtr grid, GreedyLinearObliviousTreeLearnerOptions opts) {
  return std::make_unique<GreedyLinearObliviousTreeLearner>(grid, opts);
}

inline std::unique_ptr<EmpiricalTargetFactory> createWeakTarget(double l2reg) {
  return std::make_unique<GradientBoostingWeakTargetFactory>(l2reg);
}

inline std::unique_ptr<EmpiricalTargetFactory>
createBootstrapWeakTarget(BootstrapOptions options, double l2reg) {
  return std::make_unique<GradientBoostingBootstrappedWeakTargetFactory>(
      options, l2reg);
}

int main(int /*argc*/, char* argv[]) {
  auto start = std::chrono::system_clock::now();
  auto params = readJson(argv[1]);
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

  experiments::LinearTreesBoosterOptions opts = experiments::LinearTreesBoosterOptions::fromJson(params);

  Boosting boosting(
      opts.boostingCfg, createBootstrapWeakTarget(opts.boostrapOpts, opts.greedyLinearTreesOpts.l2reg),
      createWeakLearner(grid, opts.greedyLinearTreesOpts));

  std::string target = params.value("target", "mse");
  std::unique_ptr<Target> objective;
  auto metricsCalcer = std::make_shared<BoostingMetricsCalcer>(test);
  if (target == "mse") {
    metricsCalcer->addMetric(L2(test), "l2_test");
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
