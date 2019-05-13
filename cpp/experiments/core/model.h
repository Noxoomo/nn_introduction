#include <utility>

#include <utility>

#pragma once

#include "tensor_pair_dataset.h"

#include <torch/torch.h>
#include <memory>

namespace experiments {

  class Model : public torch::nn::Module {
  public:
    virtual torch::Tensor  forward(torch::Tensor x) = 0;

    // WTF torch, this should be default behaviour
    void train(bool on = true) override {
      for (auto &param : parameters()) {
        param.set_requires_grad(on);
      }
      torch::nn::Module::train(on);
    }
  };

  using ModelPtr = std::shared_ptr<Model>;

  class Classifier : public Model {
  public:

      explicit Classifier(ModelPtr classifier) {
          classifier_ = register_module("classifier_", std::move(classifier));
      }

      explicit Classifier(ModelPtr classifier, ModelPtr baseline) {
          classifier_ = register_module("classifier_", std::move(classifier));
          baseline_ = register_module("baseline_", std::move(baseline));
          classifierScale_ = register_parameter("scale_", torch::ones({1}, torch::kFloat32));
      }

      virtual ModelPtr classifier() {
          return classifier_;
      }

      virtual ModelPtr baseline() {
          return baseline_;
      }

      virtual void enableBaselineTrain(bool flag) {
          if (baseline_) {
              baseline_->train(flag);
              classifierScale_.set_requires_grad(flag);
          }
      }
      torch::Tensor forward(torch::Tensor x) override;

  private:
      ModelPtr classifier_;
      ModelPtr baseline_;
      torch::Tensor classifierScale_;

  };



  using ClassifierPtr = std::shared_ptr<Classifier>;



  class ConvModel : public Model {
  public:
    virtual ModelPtr conv() = 0;

    virtual ClassifierPtr classifier() = 0;

    virtual torch::Tensor forward(torch::Tensor x) override;

    void train(bool on = true) override;
  };

  using ConvModelPtr =  std::shared_ptr<ConvModel>;

  class CompositionalModel : public experiments::ConvModel {
  public:
    CompositionalModel(experiments::ModelPtr first,
                       experiments::ClassifierPtr second) {
      first_ = register_module("first_", std::move(first));
      second_ = register_module("second_", std::move(second));
    }

    virtual ModelPtr conv() override  {
        return first_;
    }

    ClassifierPtr classifier() override {
        return second_;
    }

    virtual void train(bool on = true) override {
          first_->train(on);
        second_->train(on);
    }

  private:
    experiments::ModelPtr first_;
    experiments::ClassifierPtr second_;
  };

  class LinearCifarClassifier : public experiments::Model {
  public:
    LinearCifarClassifier(int dim);

    torch::Tensor forward(torch::Tensor x) override;

    ~LinearCifarClassifier() override = default;

  private:
    torch::nn::Linear fc1_{nullptr};
  };

    class SigmoidLinearCifarClassifier : public experiments::Model {
    public:
        SigmoidLinearCifarClassifier(int dim);

        torch::Tensor forward(torch::Tensor x) override;

        ~SigmoidLinearCifarClassifier() override = default;

    private:
        torch::nn::Linear fc1_{nullptr};
    };


    class Bias : public experiments::Model {
    public:
        Bias(int dim);

        torch::Tensor forward(torch::Tensor x) override;

        ~Bias() override = default;

    private:
        torch::Tensor bias_;
    };
}

inline experiments::ModelPtr makeCifarLinearClassifier(int inputDim) {
    return std::make_shared<experiments::LinearCifarClassifier>(inputDim);
}

inline experiments::ModelPtr makeCifarBias() {
    return std::make_shared<experiments::Bias>(10);
}

template <class Impl, class... Args>
inline experiments::ClassifierPtr makeClassifier(Args&&... args) {
    return std::make_shared<experiments::Classifier>(std::make_shared<Impl>(std::forward<Args>(args)...));
}

template <class Impl, class... Args>
inline experiments::ClassifierPtr makeClassifierWithBaseline(experiments::ModelPtr baseline, Args&&... args) {
    return std::make_shared<experiments::Classifier>(std::make_shared<Impl>(std::forward<Args>(args)...), baseline);
}
