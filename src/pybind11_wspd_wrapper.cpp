#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "wsp.h"

 

namespace py = pybind11;


PYBIND11_MODULE(wspd, m)
{
    
    m.doc() = "Well-separated pair decomposition (WSPD) plugin"; 
    m.def("build_wspd", &run_wspd, "Well Separated Pair Decomposition (WSPD): The build_wspd() function computes WSPD following the original Callahan and Kosaraju O(n log(n)) algorithm. \n Parameters: \n    \t arg0: int  - the size of the dataset  \n  \t arg1: int - the dimension of the dataset \n \t arg2: float - the separation constant S \n  \t arg3: List[point] - list of input points. Note that point is an internal wspd point (wspd.point). \n \n Output: The function outputs the Python list of tuples. Each tuple represents the WSPD realization link, sometimes called the dumbbell. ");

    m.def("build_wspd_tup_np",
        [](int num, int dim, double sep_const, vector<point>& pts) {
            auto dumbbells = run_wspd(num, dim, sep_const, pts);
            vector<py::array_t<int>> a_arrays, b_arrays;
            a_arrays.reserve(dumbbells.size());
            b_arrays.reserve(dumbbells.size());
            for (auto& db : dumbbells) {
                a_arrays.push_back(py::array_t<int>(
                    {(py::ssize_t)db.first.size()}, db.first.data()));
                b_arrays.push_back(py::array_t<int>(
                    {(py::ssize_t)db.second.size()}, db.second.data()));
            }
            return py::make_tuple(a_arrays, b_arrays);
        },
        "Well Separated Pair Decomposition (WSPD) with NumPy output: "
        "build_wspd_tup_np() computes WSPD and returns a tuple of two lists of 1-D NumPy arrays. "
        "The i-th array in each list contains the point indices for the A (resp. B) set of the i-th well-separated pair. "
        "\n Parameters: "
        "\n    \t arg0: int  - the size of the dataset "
        "\n  \t arg1: int - the dimension of the dataset "
        "\n \t arg2: float - the separation constant S "
        "\n  \t arg3: List[point] - list of input points (wspd.point). "
        "\n Output: tuple[list[np.ndarray], list[np.ndarray]]");

    m.def("build_wspd_flat_np",
        [](int num, int dim, double sep_const, vector<point>& pts) {
            auto dumbbells = run_wspd(num, dim, sep_const, pts);

            // Compute total sizes for flat arrays
            size_t a_total = 0, b_total = 0;
            for (auto& db : dumbbells) {
                a_total += db.first.size();
                b_total += db.second.size();
            }

            // Allocate output arrays
            py::array_t<int> a_flat(a_total);
            py::array_t<int> b_flat(b_total);
            py::array_t<int> a_offsets(dumbbells.size());
            py::array_t<int> b_offsets(dumbbells.size());

            int* a_data = a_flat.mutable_data();
            int* b_data = b_flat.mutable_data();
            int* a_off  = a_offsets.mutable_data();
            int* b_off  = b_offsets.mutable_data();

            size_t a_pos = 0, b_pos = 0;
            for (size_t i = 0; i < dumbbells.size(); ++i) {
                a_off[i] = static_cast<int>(a_pos);
                b_off[i] = static_cast<int>(b_pos);
                for (int idx : dumbbells[i].first)  a_data[a_pos++] = idx;
                for (int idx : dumbbells[i].second) b_data[b_pos++] = idx;
            }

            return py::make_tuple(a_flat, b_flat, a_offsets, b_offsets);
        },
        "Well Separated Pair Decomposition (WSPD) with flat NumPy output: "
        "build_wspd_flat_np() computes WSPD and returns a tuple of four 1-D NumPy integer arrays "
        "(a_flat, b_flat, a_offsets, b_offsets). "
        "a_flat and b_flat are the concatenations of the A and B point-index arrays across all pairs. "
        "a_offsets[i] and b_offsets[i] give the start position of the i-th pair within a_flat and b_flat, respectively. "
        "\n Parameters: "
        "\n    \t arg0: int  - the size of the dataset "
        "\n  \t arg1: int - the dimension of the dataset "
        "\n \t arg2: float - the separation constant S "
        "\n  \t arg3: List[point] - list of input points (wspd.point). "
        "\n Output: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]");

    py::class_<point>(m, "point", "Class point is an internal point representation. It accepts the Python list of floats as point coordinates. Attribute coord can be used to fetch point coordinates when needed.")
        .def(py::init<vector<double>&>())
        .def("coord", &point::coordinates);
}
