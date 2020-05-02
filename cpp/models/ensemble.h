#pragma once

#include "model.h"

class Ensemble : public Stub<Model, Ensemble> {
public:

    Ensemble(std::vector<ModelPtr>&& models)
    : Stub<Model, Ensemble>(
        models.front()->xdim(),
        models.front()->ydim())
    , models_(std::move(models)) {

    }

    Ensemble(const Ensemble& other, double scale)
    : Stub<Model, Ensemble>(other.xdim(), other.ydim())
    , models_(other.models_)
    , scale_(other.scale_ * scale){
    }

    void appendTo(const Vec& x, Vec to) const override;

    void appendToDs(const DataSet& ds, Mx to) const override {
        for (const auto& model : models_) {
            model->append(ds, to);
        }
        if (scale_ != 1.0) {
            to *= scale_;
        }
    }

    void applyToDs(const DataSet& ds, Mx to) const override {
        for (const auto& model : models_) {
            model->append(ds, to);
        }
        if (scale_ != 1.0) {
            to *= scale_;
        }
    }

    double value(const Vec& x) override {
        double res = 0;
        for (auto& model : models_) {
            res += model->value(x);
        }
        return res;
    }

    void grad(const Vec& x, Vec to) override {
        for (auto& model : models_) {
            model->grad(x, to);
        }
    }

    int64_t size() const {
        return models_.size();
    }

    template <typename TVisitor>
    ModelPtr visitModels(TVisitor visitor) const {
        for (const auto& model : models_) {
            visitor(model);
        }
    }

private:
    std::vector<ModelPtr> models_;
    double scale_ = 1.0;
};
