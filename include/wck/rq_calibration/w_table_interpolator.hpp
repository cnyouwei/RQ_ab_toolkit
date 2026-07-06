#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace wck {

class WTableInterpolator {
public:
    static WTableInterpolator from_matrix_csv(const std::filesystem::path& path);

    double w(double c, double t) const;

    [[nodiscard]] const std::vector<double>& c_grid() const noexcept { return c_grid_; }
    [[nodiscard]] const std::vector<double>& t_grid() const noexcept { return t_grid_; }

private:
    struct RowModel {
        std::vector<double> y_positive{};
        std::vector<double> slopes{};
    };

    std::vector<double> c_grid_{};
    std::vector<double> t_grid_{};
    std::vector<std::vector<double>> w_matrix_{};

    std::vector<double> t_positive_{};
    std::vector<double> x_log_positive_{};
    std::vector<RowModel> row_models_{};

    double c_min_ = 0.0;
    double c_max_ = 0.0;
    double c_tail_scale_ = 1.0;
    double t_min_positive_ = 0.0;
    double t_max_positive_ = 0.0;

    void validate_inputs() const;
    void enforce_properties();
    void build_row_models();
    double interp_row_t(std::size_t row_index, double t) const;
};

}  // namespace wck

