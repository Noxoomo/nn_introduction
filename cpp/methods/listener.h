#pragma once

#include <core/object.h>
#include <core/func.h>
#include <vector>
#include <memory>

template <class T>
class Listener : public Object {
public:
    virtual void operator()(const T& event) = 0;
};

template <class T>
class ListenersHolder : public Object {
public:
    using Inner = Listener<T>;

    void addListener(SharedPtr<Inner> listener) {
        listeners_.push_back(listener);
    }

protected:

    void invoke(const T& event) const {
        for (int64_t i = 0; i < listeners_.size(); ++i) {
            if (!listeners_[i].expired()) {
                SharedPtr<Inner> ptr = listeners_[i].lock();
                (*ptr)(event);
            }
        }
    }

private:

    std::vector<std::weak_ptr<Inner>> listeners_;
};


class BoostingMetricsCalcer : public Listener<Model> {
public:

    BoostingMetricsCalcer(const DataSet& ds)
    : ds_(ds)
    , cursor_(ds.target().dim(), 1) {

    }

    void operator()(const Model& model) override {
        model.append(ds_, cursor_);
        if (iter_ % 1 == 0) {
            std::cout << "iter " << iter_<<": ";
            for (int32_t i = 0; i < metrics_.size(); ++i) {
                double val = metrics_[i]->value(cursor_);
                if (val < bestValue_) {
                    bestValue_ = val;
                    bestIter_  = iter_;
                }
                std::cout << metricName[i] << "=" << val << ", best: (" << bestValue_ << ", " << bestIter_ << ")";
                if (i + 1 != metrics_.size()) {
                    std::cout << "\t";
                }
            }
            std::cout << std::endl;
        }
        ++iter_;
    }

    void addMetric(const Func& func, const std::string& name) {
        metrics_.push_back(func);
        metricName.push_back(name);
    }
private:
    std::vector<SharedPtr<Func>> metrics_;
    std::vector<std::string> metricName;
    const DataSet& ds_;
    Mx cursor_;
    int32_t iter_ = 0;

    double bestValue_ = 1e9;
    int bestIter_ = 0;
};

class BoostingFitTimeTracker : public Listener<Model> {
public:
    using time_point = std::chrono::steady_clock::time_point;

    explicit BoostingFitTimeTracker(int avgWindow = 50)
            : avgWindow_(avgWindow) {
        fitTimes_.push_back(std::chrono::steady_clock::now());
    }

    void operator()(const Model& model) override {
        fitTimes_.push_back(std::chrono::steady_clock::now());
        auto avgTimeLast = getAvgTime(1);
        auto avgTimeWindow = getAvgTime(avgWindow_);
        auto avgTimeAll = getAvgTime(fitTimes_.size());
        std::cout << "Fit on iter #" << iter_ << " time [ms]: [" << avgTimeLast << ", "
                << avgTimeWindow << "(" << avgWindow_ << "), " << avgTimeAll << "]" << std::endl;
        ++iter_;
    }

private:

    double getAvgTime(int window) {
        double totalTime = 0;
        int realWindowSize = 0;

        int start = (int)fitTimes_.size() - 1;
        int end = std::max(0, (int)fitTimes_.size() - window - 1);
        for (int i = start; i > end; --i, ++realWindowSize) {
            const auto& startTime = fitTimes_[i - 1];
            const auto& endTime = fitTimes_[i];
            totalTime += std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();
        }

        if (realWindowSize == 0) {
            return 0.;
        }

        return totalTime / realWindowSize;
    }

private:
    std::vector<time_point> fitTimes_;
    int avgWindow_;
    int iter_ = 0;
};


class IterPrinter : public Listener<Model> {
public:

    IterPrinter() {

    }

    void operator()(const Model& model) override {
        if (iter_ % 10 == 0) {
            std::cout << "iter " << iter_<<std::endl;
        }
        ++iter_;
    }

private:
    int32_t iter_ = 0;
};
