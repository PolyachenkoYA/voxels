//
// Created by ypolyach on 10/27/21.
//

//#include <pybind11/pybind11.h>
//#include <pybind11/stl.h>

//#include <ctime>
#include "voxels.hpp"

namespace py = pybind11;

//PYBIND11_MODULE(lattice_gas_tmp3, m)
//PYBIND11_MODULE(lattice_gas_tmp2, m)
//PYBIND11_MODULE(lattice_gas_tmp1, m)
PYBIND11_MODULE(voxels, m)
{
	/*
	 * py::tuple run_bruteforce(py_c_arr(double) rho_init_state, py_c_arr(double) Rs_init, py_c_arr(long) p_indpos_arr_new,
						 py_c_arr(int) droplet_on_inds_init,
						 double dt, long Nt_max,
						 long thermo_stride, long Rs_stride, long rho_stride,
						 py_c_arr(long) N_L_new, double l_voxel,
						 double ln_k0_off, double kA_off,
						 py_c_arr(double) ln_k0_on, py_c_arr(double) kA_on,
						 double R_0, double rho_coex, double rho_l, double R_S0, double S0, double D_coef,
						 py_c_arr(long) p_high_ind_arr_new, py_c_arr(long) NHhigh_arr_new,
						 py_c_arr(long) p_low_ind_arr_new, py_c_arr(long) f_inv_ind_arr_new,
						 py_c_arr(double) p_table, py_c_arr(double) f_inv_table,
						 double D_Hbig, double D_H,
						 int equil_mode, int to_precomp_dist, int R_crit_mode, int PBC_mode, int accuracy_mode,
						 std::optional<int> verbose_optional)
	 */
    m.def("run_bruteforce", &run_bruteforce,
          "run_bruteforce for the voxel-s model",
		  py::arg("rho_init_state"),
		  py::arg("Rs_init"),
		  py::arg("p_indpos_arr"),
		  py::arg("droplet_on_inds_init"),
		  py::arg("dt"),
		  py::arg("Nt_max"),
		  py::arg("thermo_stride"),
		  py::arg("Rs_stride"),
		  py::arg("rho_stride"),
		  py::arg("N_L"),
          py::arg("l_voxel"),
		  py::arg("ln_k0_off"),
		  py::arg("kA_off"),
		  py::arg("ln_k0_on_table"),
		  py::arg("kA_on_table"),
		  py::arg("R_0"),
		  py::arg("rho_coex"),
		  py::arg("rho_l"),
		  py::arg("R_S0"),
		  py::arg("S0"),
		  py::arg("D_coef"),
		  py::arg("p_high_ind_arr"),
		  py::arg("NHhigh_arr"),
		  py::arg("p_low_ind_arr"),
		  py::arg("f_inv_ind_arr"),
		  py::arg("p_table"),
		  py::arg("f_inv_table"),
		  py::arg("D_Hbig"),
		  py::arg("D_H"),
		  py::arg("S_thr")=0,
		  py::arg("k_tot_thr")=0,
		  py::arg("equil_mode")=equil_mode_no_eq,
		  py::arg("to_precomp_dist")=1,
		  py::arg("R_crit_mode")=R_crit_mode_CNT,
		  py::arg("PBC_mode")=PBC_mode_yes_PBC,
		  py::arg("accuracy_mode")=accuracy_mode_excl_nucls,
		  py::arg("verbose")=py::none()
    );

// py::int_ init_rand(int my_seed)
    m.def("init_rand", &init_rand,
        "Initialize GSL rand generator",
        py::arg("seed")=time(nullptr)
    );

// py::int_ set_verbose(int new_verbose)
    m.def("set_verbose", &set_verbose,
          "Set default verbose behaviour",
          py::arg("new_verbose")
    );

// py::int_ get_verbose()
	m.def("get_verbose", &get_verbose,
		  "Get default verbose level"
		  );

// py::int_ set_verbose(int new_verbose)
	m.def("get_seed", &get_seed,
		  "Returns the current seed used for the last GSL random initiation"
	);

////	void print_possible_move_modes()
//	m.def("print_possible_move_modes", &print_possible_move_modes,
//		  "Prints the possible options for MC-moves"
//	);

//	py::dict get_equil_modes()
	m.def("get_equil_modes", &get_equil_modes,
		  "get a tuple of dicts for Equilibration modes: 1) {'name' : value}; 2) {value : 'name'}"
	);

//	py::dict get_accuracy_modes()
	m.def("get_accuracy_modes", &get_accuracy_modes,
		  "get a tuple of dicts for Accuracy modes: 1) {'name' : value}; 2) {value : 'name'}"
	);

//	py::dict get_Rcrit_modes()
	m.def("get_Rcrit_modes", &get_Rcrit_modes,
		  "get a tuple of dicts for Rcrit modes: 1) {'name' : value}; 2) {value : 'name'}"
	);

//	py::dict get_p_gen_modes()
	m.def("get_p_gen_modes", &get_p_gen_modes,
		  "get a tuple of dicts for p_gen modes: 1) {'name' : value}; 2) {value : 'name'}"
	);

//	py::dict get_rho_gen_modes()
	m.def("get_rho_gen_modes", &get_rho_gen_modes,
		  "get a tuple of dicts for p_gen modes: 1) {'name' : value}; 2) {value : 'name'}"
	);

}
