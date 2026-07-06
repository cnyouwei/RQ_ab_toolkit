#include "wck/rq_calibration/w_table_interpolator.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <fstream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace wck {

namespace {

std::string trim_copy(const std::string& text) {
    std::size_t begin = 0U;
    while (begin < text.size() && std::isspace(static_cast<unsigned char>(text[begin])) != 0) {
        ++begin;
    }

    std::size_t end = text.size();
    while (end > begin && std::isspace(static_cast<unsigned char>(text[end - 1U])) != 0) {
        --end;
    }

    return text.substr(begin, end - begin);
}

std::vector<std::string> split_csv_fields(const std::string& line) {
    std::vector<std::string> fields{};
    std::size_t start = 0U;
    while (start <= line.size()) {
        const std::size_t comma = line.find(',', start);
        const std::size_t end = (comma == std::string::npos) ? line.size() : comma;
        fields.push_back(line.substr(start, end - start));
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1U;
    }
    return fields;
}

double parse_double_strict(const std::string& text, const std::string& context) {
    std::size_t parsed = 0U;
    double value = 0.0;
    try {
        value = std::stod(trim_copy(text), &parsed);
    } catch (...) {
        throw std::invalid_argument("invalid numeric value for " + context + ": " + text);
    }

    const std::string trimmed = trim_copy(text);
    if (parsed != trimmed.size()) {
        throw std::invalid_argument("invalid numeric value for " + context + ": " + text);
    }
    if (!std::isfinite(value)) {
        throw std::invalid_argument("non-finite numeric value for " + context + ": " + text);
    }
    return value;
}

std::vector<double> pchip_slopes(const std::vector<double>& x, const std::vector<double>& y) {
    const std::size_t n = x.size();
    if (n != y.size()) {
        throw std::invalid_argument("pchip_slopes: x/y size mismatch");
    }
    if (n < 2U) {
        throw std::invalid_argument("pchip_slopes: at least two points required");
    }
    if (n == 2U) {
        const double slope = (y[1] - y[0]) / (x[1] - x[0]);
        return std::vector<double>{slope, slope};
    }

    std::vector<double> h(n - 1U, 0.0);
    std::vector<double> delta(n - 1U, 0.0);
    for (std::size_t i = 0U; i + 1U < n; ++i) {
        h[i] = x[i + 1U] - x[i];
        if (!(h[i] > 0.0)) {
            throw std::invalid_argument("pchip_slopes: x must be strictly increasing");
        }
        delta[i] = (y[i + 1U] - y[i]) / h[i];
    }

    std::vector<double> m(n, 0.0);
    for (std::size_t k = 1U; k + 1U < n; ++k) {
        if (delta[k - 1U] == 0.0 || delta[k] == 0.0) {
            m[k] = 0.0;
            continue;
        }
        const bool sign_diff = (delta[k - 1U] > 0.0) != (delta[k] > 0.0);
        if (sign_diff) {
            m[k] = 0.0;
            continue;
        }

        const double w1 = 2.0 * h[k] + h[k - 1U];
        const double w2 = h[k] + 2.0 * h[k - 1U];
        m[k] = (w1 + w2) / (w1 / delta[k - 1U] + w2 / delta[k]);
    }

    double m0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
    if ((m0 > 0.0) != (delta[0] > 0.0)) {
        m0 = 0.0;
    } else if (((delta[0] > 0.0) != (delta[1] > 0.0)) && std::abs(m0) > std::abs(3.0 * delta[0])) {
        m0 = 3.0 * delta[0];
    }
    m[0] = m0;

    const std::size_t last = n - 1U;
    double mn = ((2.0 * h[last - 1U] + h[last - 2U]) * delta[last - 1U] - h[last - 1U] * delta[last - 2U])
        / (h[last - 1U] + h[last - 2U]);
    if ((mn > 0.0) != (delta[last - 1U] > 0.0)) {
        mn = 0.0;
    } else if (((delta[last - 1U] > 0.0) != (delta[last - 2U] > 0.0))
               && std::abs(mn) > std::abs(3.0 * delta[last - 1U])) {
        mn = 3.0 * delta[last - 1U];
    }
    m[last] = mn;
    return m;
}

double pchip_eval_scalar(const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& m, double xq) {
    const std::size_t n = x.size();
    if (n < 2U) {
        return y.front();
    }
    if (xq <= x.front()) {
        return y.front();
    }
    if (xq >= x.back()) {
        return y.back();
    }

    std::size_t i = static_cast<std::size_t>(std::upper_bound(x.begin(), x.end(), xq) - x.begin());
    if (i == 0U) {
        i = 1U;
    }
    --i;
    if (i + 1U >= n) {
        i = n - 2U;
    }

    const double h = x[i + 1U] - x[i];
    const double s = (xq - x[i]) / h;

    const double h00 = 2.0 * s * s * s - 3.0 * s * s + 1.0;
    const double h10 = s * s * s - 2.0 * s * s + s;
    const double h01 = -2.0 * s * s * s + 3.0 * s * s;
    const double h11 = s * s * s - s * s;

    return h00 * y[i] + h10 * h * m[i] + h01 * y[i + 1U] + h11 * h * m[i + 1U];
}

}  // namespace

WTableInterpolator WTableInterpolator::from_matrix_csv(const std::filesystem::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("table not found: " + path.string());
    }

    std::string header_line{};
    if (!std::getline(in, header_line)) {
        throw std::invalid_argument("matrix table has no header: " + path.string());
    }
    const std::vector<std::string> header = split_csv_fields(header_line);
    if (header.size() < 2U) {
        throw std::invalid_argument("matrix header must include c and at least one t-column");
    }
    if (trim_copy(header[0]) != "c") {
        throw std::invalid_argument("matrix header must start with 'c'");
    }

    std::vector<double> t_grid{};
    t_grid.reserve(header.size() - 1U);
    for (std::size_t j = 1U; j < header.size(); ++j) {
        t_grid.push_back(parse_double_strict(header[j], "header column t[" + std::to_string(j - 1U) + "]"));
    }

    std::vector<double> c_values{};
    std::vector<std::vector<double>> matrix{};

    std::string line{};
    std::size_t line_number = 1U;
    while (std::getline(in, line)) {
        ++line_number;
        if (trim_copy(line).empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_csv_fields(line);
        if (fields.size() != header.size()) {
            throw std::invalid_argument(
                "inconsistent row length at line " + std::to_string(line_number) + " in " + path.string());
        }

        c_values.push_back(parse_double_strict(fields[0], "c at line " + std::to_string(line_number)));
        std::vector<double> row{};
        row.reserve(t_grid.size());
        for (std::size_t j = 1U; j < fields.size(); ++j) {
            row.push_back(parse_double_strict(
                fields[j],
                "w value at line " + std::to_string(line_number) + ", column " + std::to_string(j)));
        }
        matrix.push_back(std::move(row));
    }

    if (c_values.empty()) {
        throw std::invalid_argument("matrix table must have at least one data row: " + path.string());
    }

    std::vector<std::size_t> order(c_values.size(), 0U);
    std::iota(order.begin(), order.end(), 0U);
    std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return c_values[lhs] < c_values[rhs];
    });

    WTableInterpolator out{};
    out.c_grid_.reserve(c_values.size());
    out.w_matrix_.reserve(matrix.size());
    out.t_grid_ = std::move(t_grid);
    for (std::size_t idx : order) {
        out.c_grid_.push_back(c_values[idx]);
        out.w_matrix_.push_back(std::move(matrix[idx]));
    }

    out.validate_inputs();

    if (out.t_grid_.front() > 0.0) {
        out.t_grid_.insert(out.t_grid_.begin(), 0.0);
        for (auto& row : out.w_matrix_) {
            row.insert(row.begin(), 1.0);
        }
    } else if (std::abs(out.t_grid_.front()) < 1e-14) {
        out.t_grid_.front() = 0.0;
        for (auto& row : out.w_matrix_) {
            row.front() = 1.0;
        }
    } else {
        throw std::invalid_argument("t-grid must start at t >= 0");
    }

    out.enforce_properties();
    out.build_row_models();

    out.c_min_ = out.c_grid_.front();
    out.c_max_ = out.c_grid_.back();
    out.c_tail_scale_ = std::max(1.0, 0.2 * (out.c_max_ - out.c_min_));
    out.t_min_positive_ = out.t_positive_.front();
    out.t_max_positive_ = out.t_positive_.back();

    return out;
}

void WTableInterpolator::validate_inputs() const {
    if (c_grid_.empty()) {
        throw std::invalid_argument("c-grid is empty");
    }
    if (t_grid_.empty()) {
        throw std::invalid_argument("t-grid is empty");
    }
    if (c_grid_.size() != w_matrix_.size()) {
        throw std::invalid_argument("c-grid and matrix row count mismatch");
    }
    for (const auto& row : w_matrix_) {
        if (row.size() != t_grid_.size()) {
            throw std::invalid_argument("inconsistent matrix row size");
        }
    }
    for (std::size_t i = 0U; i + 1U < c_grid_.size(); ++i) {
        if (!(c_grid_[i + 1U] > c_grid_[i])) {
            throw std::invalid_argument("c-grid must be strictly increasing");
        }
    }
    for (std::size_t i = 0U; i + 1U < t_grid_.size(); ++i) {
        if (!(t_grid_[i + 1U] > t_grid_[i])) {
            throw std::invalid_argument("t-grid must be strictly increasing");
        }
    }
}

void WTableInterpolator::enforce_properties() {
    for (auto& row : w_matrix_) {
        row[0] = 1.0;
    }
}

void WTableInterpolator::build_row_models() {
    std::vector<std::size_t> positive_indices{};
    positive_indices.reserve(t_grid_.size());
    for (std::size_t i = 0U; i < t_grid_.size(); ++i) {
        if (t_grid_[i] > 0.0) {
            positive_indices.push_back(i);
        }
    }
    if (positive_indices.empty()) {
        throw std::invalid_argument("table has no positive t grid points");
    }

    t_positive_.clear();
    x_log_positive_.clear();
    t_positive_.reserve(positive_indices.size());
    x_log_positive_.reserve(positive_indices.size());
    for (std::size_t idx : positive_indices) {
        t_positive_.push_back(t_grid_[idx]);
        x_log_positive_.push_back(std::log(t_grid_[idx]));
    }

    row_models_.clear();
    row_models_.reserve(w_matrix_.size());
    for (const auto& row : w_matrix_) {
        RowModel model{};
        model.y_positive.reserve(positive_indices.size());
        for (std::size_t idx : positive_indices) {
            model.y_positive.push_back(row[idx]);
        }

        if (model.y_positive.size() == 1U) {
            model.slopes = std::vector<double>{0.0};
        } else {
            model.slopes = pchip_slopes(x_log_positive_, model.y_positive);
        }
        row_models_.push_back(std::move(model));
    }
}

double WTableInterpolator::interp_row_t(std::size_t row_index, double t) const {
    if (!(row_index < row_models_.size())) {
        throw std::out_of_range("row index out of range in interp_row_t");
    }
    if (t <= 0.0) {
        return 1.0;
    }

    const RowModel& row = row_models_[row_index];
    if (t <= t_min_positive_) {
        const double w0 = row.y_positive.front();
        return 1.0 - (1.0 - w0) * (t / t_min_positive_);
    }
    if (t >= t_max_positive_) {
        return row.y_positive.back();
    }

    const double xq = std::log(t);
    return pchip_eval_scalar(x_log_positive_, row.y_positive, row.slopes, xq);
}

double WTableInterpolator::w(double c, double t) const {
    if (t <= 0.0) {
        return 1.0;
    }

    if (c <= c_min_) {
        const double w_edge = interp_row_t(0U, t);
        return 1.0 - (1.0 - w_edge) * std::exp((c - c_min_) / c_tail_scale_);
    }

    if (c >= c_max_) {
        const double w_edge = interp_row_t(c_grid_.size() - 1U, t);
        return w_edge * std::exp(-(c - c_max_) / c_tail_scale_);
    }

    std::size_t i = static_cast<std::size_t>(std::upper_bound(c_grid_.begin(), c_grid_.end(), c) - c_grid_.begin());
    if (i == 0U) {
        i = 1U;
    }
    --i;
    if (i + 1U >= c_grid_.size()) {
        i = c_grid_.size() - 2U;
    }

    const double c0 = c_grid_[i];
    const double c1 = c_grid_[i + 1U];
    const double w0 = interp_row_t(i, t);
    const double w1 = interp_row_t(i + 1U, t);
    const double theta = (c - c0) / (c1 - c0);
    return (1.0 - theta) * w0 + theta * w1;
}

}  // namespace wck
