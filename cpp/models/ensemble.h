#pragma once

#include "model.h"

#include <util/io.h>

class Ensemble : public Stub<Model, Ensemble> {
public:

    Ensemble(std::vector<ModelPtr> models)
    : Stub<Model, Ensemble>(
        models.front()->xdim(),
        models.front()->ydim())
    , models_(std::move(models)) {

    }

    Ensemble(std::vector<ModelPtr> models, double scale)
            : Stub<Model, Ensemble>(
            models.front()->xdim(),
            models.front()->ydim())
            , models_(std::move(models))
            , scale_(scale) {

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
    void visitModels(TVisitor visitor) const {
        for (const auto& model : models_) {
            visitor(model);
        }
    }

    template <typename TSerializer>
    void serialize(std::ostream& out, TSerializer serializer) const {
        out.write("e{", 2);
        out.write((char*)&scale_, sizeof(scale_));
        out.write("}", 1);
        for (const auto& model : models_) {
            serializer(model);
        }
    }

    template <typename TDeserializer>
    static std::shared_ptr<Ensemble> deserialize(std::istream& in, TDeserializer deserializer) {
        double scale;
        if (!couldRead(in, &scale, sizeof(scale), "e")) {
            return nullptr;
        }

        std::vector<ModelPtr> models;

        while (true) {
            auto model = deserializer();
            if (model) {
                models.emplace_back(std::move(model));
            } else {
                break;
            }
        }

        if (models.empty()) {
            return nullptr;
        }
        return std::make_shared<Ensemble>(std::move(models), scale);
    }

    template <typename TSerializer>
    void serializeLast(std::ostream& out, TSerializer serializer, int nLast = 1) const {
        if (!serializeLastCalled_) {
            serializeLastCalled_ = true;
            out.write("e{", 2);
            out.write((char*)&scale_, sizeof(scale_));
            out.write("}", 1);
        }
        for (int i = std::max(0, (int)(models_.size()) - nLast); i < (int)models_.size(); ++i) {
            serializer(models_[i]);
        }
    }

private:
    std::vector<ModelPtr> models_;
    double scale_ = 1.0;

    mutable bool serializeLastCalled_ = false;
};
