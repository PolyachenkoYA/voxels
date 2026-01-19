//
// Created by ypolyach on 10/27/21.
//

#ifndef VOXELS_VOXELS_H
#define VOXELS_VOXELS_H

#include <gsl/gsl_rng.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_randist.h>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pytypes.h>
#include <pybind11/cast.h>

#include <Python.h>
//#include <python3.8/Python.h>

#include <random>
#include <set>
#include <algorithm>
#include <type_traits>

namespace py = pybind11;
using namespace py::literals;

#define dim 3

#define equil_mode_no_eq 0

#define PBC_mode_no_PBC 0
#define PBC_mode_yes_PBC 1

#define R_crit_mode_const 0
#define R_crit_mode_CNT 1

#define accuracy_mode_0 0
#define accuracy_mode_excl_nucls 1 << 0
#define accuracy_mode_integr_outside 1 << 1

#define p_gen_mode_shuffle 1
#define p_gen_mode_1site 2
#define p_gen_mode_2neibrs 3

#define quit_status_Ntmax (1 << 0)
#define quit_status_Sthr (1 << 1)
#define quit_status_Ktot (1 << 2)

#define rho_gen_mode_RhoConst_noRs 1

#define STP assert(getchar() != 'e');

#define FOR_DIM for(int _i_dim = 0; _i_dim < dim; ++_i_dim)

#define py_c_arr(T) py::array_t<T, py::array::c_style | py::array::forcecast>
//#define py_c_arr(T) py::array_t<T>

template <typename T>
T* c_malloc(long s){ return static_cast<T*>(std::malloc(s * sizeof(T))); }

template <typename T>
void c_malloc(T** p, long s){ *p = c_malloc<T>(s); }

template <typename T>
T* c_realloc(T* src, long s){ return static_cast<T*>(std::realloc(src, s * sizeof(T))); }

template <typename T>
void c_realloc_this(T** p, long s){ *p = c_realloc(*p, s); }

template <typename T>
void c_free(T** p){
	if(p) {
		std::free(*p);
		(*p) = nullptr;
	}
}

template <typename T> __attribute__((noinline)) T sqr(T x) { return x * x; }
template <typename T> void zero_array(T* v, long N, T v0=0) { for(long i = 0; i < N; ++i) v[i] = v0; }
template <typename T> T sum_array(const T* v, long N) { T s = 0; for(long i = 0; i < N; ++i) s += v[i]; return s; }
template <typename T> T min_array(const T* v, long N) {
	T s = v[0];
	for(long i = 1; i < N; ++i) if(v[i] < s) s = v[i];
	return s;
}
template <typename T> T max_array(const T* v, long N) {
	T s = v[0];
	for(long i = 1; i < N; ++i) if(v[i] > s) s = v[i];
	return s;
}
template <typename T> T prod_array(const T* v, long N) { T s = 1; for(long i = 0; i < N; ++i) s *= v[i]; return s; }
template <typename T> char sgn(T val) { return T(0) <= val ? 1 : -1;	}

template <typename T> T powi(T x, long p) {
	T a = x;
	for(long i = 1; i < p; ++i) a *= x;
	return a;
}
template <typename T> T max(const T *v, long N) {
	T mx = v[0];
	for(long i = 0; i < N; ++i) if(mx < v[i]) mx = v[i];
	return  mx;
}

template <typename T>
void reorder(T * arr, const long *inds, long N){
	T* buf;
	c_malloc(& buf, N);
	memcpy(buf, arr, sizeof(T) * N);
	for(long i = 0; i < N; ++i) arr[i] = buf[inds[i]];
	c_free(& buf);
}

template <typename T>
struct argsort_str{
	T value;
	long index;
};
template <typename T>
int argsort_cmp(const void *a, const void *b){
	argsort_str<T> *a1 = (argsort_str<T> *)a;
	argsort_str<T> *b1 = (argsort_str<T> *)b;
	return (a1->value) > (b1->value) ? 1 : ((a1->value) < (b1->value) ? -1 : 0);
}
template <typename T> void argsort(const T *v, long ** inds, long N, int to_alloc=0) {
	if(to_alloc) c_malloc(inds, N);
	argsort_str<T> *pairs;
	c_malloc(& pairs, N);
	for(long i = 0; i < N; ++i){
		pairs[i].value = v[i];
		pairs[i].index = i;
	}
	qsort(pairs, N, sizeof(argsort_str<T>), argsort_cmp<T>);
	for(long i = 0; i < N; ++i){
		(*inds)[i] = pairs[i].index;
	}
	c_free(& pairs);
}

template <typename T> T* extract_arr_from_pybind11(py_c_arr(T) &pybind11_arr,
												   std::vector<long> size_check = {0}, int check_dim=0) {
	py::buffer_info pybind11_arr_info = pybind11_arr.request();
	if(check_dim > 0) assert(pybind11_arr_info.ndim == check_dim);
	if(size_check[0] > 0) {
		assert(size_check.size() == pybind11_arr_info.ndim);
		for(long i = 0; i < pybind11_arr_info.ndim; ++i){
			assert(pybind11_arr_info.shape[i] == size_check[i]);
		}
	}
	return static_cast<T *>(pybind11_arr_info.ptr);
}
template <typename T> T* extract_arr_from_pybind11(py_c_arr(T) &pybind11_arr,
												   long size_check, int check_dim=0) {
	return extract_arr_from_pybind11(pybind11_arr, std::vector<long>{size_check}, check_dim);
}

template <typename T>
void c_arr_to_py_arr(py_c_arr(T) *py_arr, T** c_arr, long N, int to_free=1)
{
	*py_arr = py_c_arr(T)(N);
	py::buffer_info py_arr_info = (*py_arr).request();
	T * py_arr_ptr = static_cast<T *>(py_arr_info.ptr);
	memcpy(py_arr_ptr, *c_arr, sizeof(T) * N);
	if(to_free) c_free(c_arr);
}

namespace voxels
{
	class Run_parameters;
	class System_parameters;
	class Acetylation_state;
	class Proteins_state;
	class Thermo_log;
	class Rho_log;
	class Rs_log;

//	extern std::mt19937 *gen_mt19937;
	extern gsl_rng *rng;
    extern long seed;
    extern int verbose_default;

	double k_dens_fnc(double ln_k0, double kA, double x);

	template <typename T> T md(T x, T L){ return x >= 0 ? (x < L ? x : x - L) : (L + x); }   // x mod L for x \in [-L; 2L)
	template <typename T> T mds(T x, T R){ return md(x + R, 2 * R) - R; }
	template <typename T> T mdc(T x, T R, T L){ return x >= -R ? (x < R ? x : x - L) : x + L; }   // x mod L for x \in [-3R; 3R)
	template <typename T> void mult_3d(T * r, double a){ FOR_DIM r[_i_dim] *= a; }
	template <typename T> void plus_3d(const T * r1, const T * r2, T * res){ FOR_DIM res[_i_dim] = r1[_i_dim] + r2[_i_dim]; }
	template <typename T> void minus_3d(const T * r1, const T * r2, T * res){ FOR_DIM res[_i_dim] = r1[_i_dim] - r2[_i_dim]; }
	template <typename T> void get_CoM(const T * r1, double m1, const T * r2, double m2, T * r_com){
		FOR_DIM r_com[_i_dim] = (r1[_i_dim] * m1 + r2[_i_dim] * m2) / (m1 + m2);
	}
	template <typename T> T dist2_3d(const T * r1, const T * r2){
		T dist2 = 0;
		FOR_DIM dist2 += sqr(r1[_i_dim] - r2[_i_dim]);
		return dist2;
	}
	template <typename T> void md_3d(const T * L, const T * r, T * res) { FOR_DIM res[_i_dim] = md(r[_i_dim], L[_i_dim]); }
	template <typename T> void minus_3d_pbc(const T * R, const T * L, const T * r1, const T * r2, T * res){ FOR_DIM res[_i_dim] = mdc(r1[_i_dim] - r2[_i_dim], R[_i_dim], L[_i_dim]); }
	template <typename T> void get_CoM_pbc(const T * R, const T * L, const T * r1, double m1, const T * r2, double m2, T * r_com){
		T dr[dim];   // dim x (N_points - 1)
		minus_3d_pbc(R, L, r2, r1, dr);   // move all points to the 1st-partcle-origin
		m2 /= m1;   // normalize by the mass of the particles the frame of which we are using
		mult_3d(dr, m2);   // scale each dr by m_i / m_1
		mult_3d(dr, 1.0 / (1.0 + m2));   // sum all scaled dr-s and scale by 1/(1 + sum{m_i / m_1})
		memcpy(r_com, r1, sizeof(T) * dim);   // count r_com from r1
		plus_3d(r_com, dr, r_com);   // get CoM in the frame of r1
		md_3d(L, r_com, r_com);   // move back to the lab PBC frame
	}
	template <typename T> T dist2_3d_pbc(const T * R, const T * L, const T * r1, const T * r2){
		T dist2 = 0;
		double dx;
		FOR_DIM{
			dx = abs(r1[_i_dim] - r2[_i_dim]);
			if(dx > R[_i_dim]) dx -= L[_i_dim];
			dist2 += dx * dx;
		}

//		FOR_DIM dist2 += sqr(mdc(r1[_i_dim] - r2[_i_dim], R[_i_dim], L[_i_dim]));
		return dist2;
	}
	void indpos2pos(long indpos, const long *N_L, long *pos);
	void indpos2pos(long indpos, const long *N_L, double *pos, double l);
	long pos2indpos(const long *pos, const long *N_L);
	long pos2indpos(long x, long y, long z, const long *N_L);
	long pos2indpos(const double *pos, const long *N_L, double l);

	class Log{
	public:
		Log() = default;
		Log(long len_new, long max_len_new, long stride_new) :
			len(len_new), max_len(max_len_new), stride(stride_new)
		{}

		long len{0}, max_len{0};
		long stride{1};

		void clear_lens(){ this->len = 0; this->max_len = 0; };
		virtual void print_params();
	};

	class Rho_log : public Log{
	public:
		Rho_log() = default;
		Rho_log(long stride_new, long state_size_new):
				Log(0, 0, stride_new), state_size(state_size_new)
		{}
		Rho_log(long len_new, long max_len_new, long stride_new, long state_size_new):
				Log(len_new, max_len_new, stride_new), state_size(state_size_new)
		{ this->alloc(); }

		double *states{nullptr};
		double *time{nullptr};
		long state_size{1};

		void alloc(long max_len_new);
		void alloc();
		void clear();
		void resize(long max_len_new);
		void add_record(double time_new, double * state_new, int verbose);
		void print_params() override;
		[[nodiscard]] double * get_ith_state(long i) const{ return &(this->states[this->state_size * i]); }

		~Rho_log(){ this->clear(); }
	};

	class Rs_log : public Log{
	public:
		Rs_log() = default;
		explicit Rs_log(long stride_new):
			Log(0, 0, stride_new), N_total(0), max_N_total(0)
		{}
		Rs_log(long len_new, long max_len_new, long stride_new, long N_total_new, long max_N_total_new):
			Log(len_new, max_len_new, stride_new), N_total(N_total_new), max_N_total(max_N_total_new)
		{}

		double *Rs{nullptr};
		long *indpos{nullptr};
		int *is_on{nullptr};
		double *time{nullptr};
		long *N_Rs{nullptr};   // The number of Rs>0 for each frame. So this marks how Rs and indpos can be decoded into separate frames
		long N_total{0}, max_N_total{0};

		void alloc(long max_len_new, long max_N_total_new);
		void alloc();
		void clear();
		void resize_Rs(long max_N_total_new);
		void resize_NR(long max_len_new);
		void add_record(double time_new, long N_R_new, const double * Rs_new, const long* indpos_new, const int* is_on_new, int verbose);
		void print_params() override;

		~Rs_log(){ this->clear(); }
	};

	class Thermo_log : public Log{
	public:
		Thermo_log() = default;
		explicit Thermo_log(long stride_new):
			Log(0, 0, stride_new)
		{}
		Thermo_log(long len_new, long max_len_new, long stride_new):
			Log(len_new, max_len_new, stride_new)
		{}

		double *S{nullptr};
		double *rate_off_total{nullptr};
		double *rate_on_total{nullptr};
		long *N_droplets{nullptr};
		long *N_on_droplets{nullptr};
		long *N_merges{nullptr};
		long *timestep{nullptr};
		double *time{nullptr};
		double *Va{nullptr};
		double *Vb{nullptr};
		double *Vc{nullptr};

		void alloc(long max_len_new);
		void clear();
		void resize(long max_len_new);
		void add_record(double S_new, long N_droplets_new, long N_on_droplets_new, long N_merges_new, long timestep_new,
						double time_new,
						double rate_off_total_new, double rate_on_total_new, double Va_new, double Vb_new, double Vc_new,
						int verbose);

		void print_params() override;
		void print();

		~Thermo_log();
	};

	class Proteins_state{
	public:
		long N_L[dim]{1,1,1};
		long N_vox{1};
		int to_comp_dots{0};
		double rho_inf{0};
		double rho_inf_dot{0};
		double *rho_state{nullptr};
		double *invlogS_state{nullptr};
		double *rates_on{nullptr};
		double *rates_off{nullptr};
		double *Rs{nullptr};
		long N_droplets{0};   // size of the array Rs_indpos
		long N_droplet_max{0};   // to realloc on droplet creation
		long *Rs_indpos{nullptr};   // In principle, this should be a UnorderedMultiSet. It will make removing elements easier
		int *droplet_on_inds{nullptr};
		double *R2{nullptr};
		double *R3{nullptr};
		double *dR{nullptr};
		double *Rs_dot{nullptr};
		double *Sj{nullptr};
		double *Sj_dot{nullptr};
		int alloced_Rs{0}, alloced_rho{0};
		double Va{0}, Vb{0}, Vb0{0}, Vc{0};
		double Va_dot{0}, Vb_dot{0}, Vb0_dot{0}, Vc_dot{0};
		double k_on_total{0}, k_off_total{0};

		Proteins_state()=default;
		explicit Proteins_state(const long* N_L_new, long N_droplets_new, int to_comp_dots_new, int to_alloc=0);
		explicit Proteins_state(py_c_arr(long) & N_L_new, long N_droplets_new, int to_comp_dots_new, int to_alloc=0);
		Proteins_state(const long* N_L_new, double *rho_state_new, double* Rs_new, const int *droplet_on_inds_new,
					   int to_comp_dots_new, int to_alloc=0);
		Proteins_state(const long* N_L_new, int rho_gen_mode, double rho_init, int to_comp_dots_new);
		explicit Proteins_state(Proteins_state * src_state) { this->copy_from(src_state, 1); }

		void copy_from(Proteins_state * src, int to_alloc=0);

		void change_state_size(const long* N_L_new);
		void alloc();
		void alloc_Rs();
		void realloc_Rs(long len_new);
		void clear();
		void clear_Rs();
		void print_params();

		void add_droplet(double R, long indpos, int is_on);

		[[nodiscard]] double protein_total_amount() const;
		[[nodiscard]] double protein_total_amount(double l) const;
		void update_Va();
		void update_Vb(System_parameters & sys_prms);
		void update_Vbs(System_parameters & sys_prms);
		void update_Vdots(System_parameters & sys_prms);
		void update_R_powers();
		void update_logS(double rho_coex, double rho_thr);
		void update_dR(System_parameters & sys_prms);
		void update_Sj(System_parameters & sys_prms);
		[[nodiscard]] double comp_Va() const;
		double comp_Vb(System_parameters & sys_prms) const;
		void comp_Vb_s(System_parameters & sys_prms, double *Vb0_local, double *Vc_local) const;
		double comp_Sj(System_parameters & sys_prms, long j) const;
		void update_rho_inf(System_parameters & sys_prms);
		void update_Rdot(System_parameters & sys_prms);
		void update_rates(System_parameters & sys_prms, Acetylation_state & acetyl_state);
		void thermo_log_dump(long Nt, long N_merges, double time, Thermo_log & thermo_log, System_parameters & sys_prms);
		void Rs_log_dump(double time, Rs_log & rs_log, System_parameters & sys_prms);
		void Rho_log_dump(double time, Rho_log & rho_log, System_parameters & sys_prms);
		double get_good_dt(System_parameters & sys_prms);

		void step_Rs(double dt);
		void generate_new_nuclei(double R, double dt);
		long merge_existing_nuclei(System_parameters & sys_prms, Acetylation_state & acetyl_state);

		void update_rho_state(System_parameters & sys_prms);

		void generate(int rho_gen_mode, double rho_init);

		int run_state(long *Nt,
					  Acetylation_state & acetylation_state,
					  System_parameters & sys_prms,
					  Run_parameters & run_prms,
					  Thermo_log & thermo_log,
					  Rs_log & rs_log,
					  Rho_log & rho_log);

		~Proteins_state();
	};

	class Run_parameters{
	public:
		Run_parameters()=default;
		Run_parameters(double dt_new, long thermo_stride_new, long Rs_stride_new, long rho_stride_new,
					   long Nt_max_new, int equil_mode_new, double S_thr_new, double ktot_thr_new)
				: dt(dt_new), thermo_stride(thermo_stride_new), Rs_stride(Rs_stride_new), rho_stride(rho_stride_new),
				  equil_mode(equil_mode_new), Nt_max(Nt_max_new), S_thr(S_thr_new), k_tot_thr(ktot_thr_new)
		{}

		double dt{1e-3};
		long thermo_stride{1}, Rs_stride{1}, rho_stride{1};
		int equil_mode{equil_mode_no_eq};
		long Nt_max{0};
		double S_thr{0};
		double k_tot_thr{0};

		void print_params() const;
	};

	class System_parameters{
	public:
		System_parameters()=default;
		System_parameters(const long* N_L_new, int PBC_mode_new, double rho_coex_new, double rho_l_new, double R_S0_new,
						  double S0_new, double D_new, double l_voxel_new, double R_0_new, double ln_k0_off_new,
						  double kA_off_new, int to_precomp_dist, int R_crit_mode_new, int accuracy_mode_new,
						  int verbose_new);
		System_parameters(py_c_arr(long) & N_L_new, int PBC_mode_new, double rho_coex_new, double rho_l_new,
						  double R_S0_new, double S0_new, double D_new, double l_voxel_new, double R_0_new,
						  double ln_k0_off_new, double kA_off_new, int to_precomp_dist, int R_crit_mode_new,
						  int accuracy_mode_new, int verbose_new);
		~System_parameters(){ if(inv_pdist) std::free(inv_pdist); }

		double rho_coex{1e-4}, rho_l{1}, R_S0{1e2}, S0{2}, D{1}, l_voxel{0}, R_0{0}, v_voxel{1}, V{1}, Rmax{1};
		long N_vox{1};
		long N_L[dim]{1, 1, 1};
		double ln_k0_off{0}, kA_off{1};
		int PBC_mode{0};
		double R[dim]{1,1,1};
		double L[dim]{2,2,2};
		double * inv_pdist{nullptr};
		int accuracy_mode{accuracy_mode_0};
		int R_crit_mode{R_crit_mode_const};
		double protein_total_amount{0};
		int verbose{0};

		void print_params();
		void change_system_size(const long* N_L_new, int to_precomp_dist);
		[[nodiscard]] double get_invdist_ordered(long i, long j) const{ return this->inv_pdist[i * this->N_vox + j - (i+2)*(i+1)/2]; } // i < j
		[[nodiscard]] double get_invdist(long i, long j) const{ if(i > j) std::swap(i, j); return this->get_invdist_ordered(i, j); }
		void set_invdist(long i, long j, double d){ if(i > j) std::swap(i, j); this->set_invdist_ordered(i, j, d); }
		void set_invdist_ordered(long i, long j, double d){ this->inv_pdist[i * this->N_vox + j - (i+2)*(i+1)/2] = d; }
	};

	class Acetylation_state {
		/*
		 * This class does not store the actual data, is just stores the pointers to the data.
		 * This way there is not need to copy the arrays passed from Python.
		 */
	public:
		Acetylation_state()=default;
		Acetylation_state(long N_active_sites_new, double D_Hbig_new, double D_H_new, long N_vox_new,
						  long *indpos_arr_new,
						  long *NHhigh_arr_new, long *p_high_ind_arr_new, long *p_low_ind_arr_new,
						  long *f_inv_ind_arr_new, double *_ln_k0_on, double *_kA_on, double *_p_arr,
						  double *_f_inv_arr, long N_p_new, long N_f_new);
		Acetylation_state(long N_active_sites_new, double D_Hbig_new, double D_H_new, long N_vox_new,
						  py_c_arr(long) & indpos_arr_new,
						  py_c_arr(long) & NHhigh_arr_new, py_c_arr(long) & p_high_ind_arr_new,
						  py_c_arr(long) & p_low_ind_arr_new, py_c_arr(long) & f_inv_ind_arr_new,
						  py_c_arr(double) & _ln_k0_on, py_c_arr(double) & _kA_on,
						  py_c_arr(double) & _p_arr, py_c_arr(double) & _f_inv_arr);
		Acetylation_state(const long *N_L, long N_active_sites_new, double D_Hbig_new, double D_H_new, long N_vox_new,
						  int p_gen_mode, long p_high_ind, long NH_high, long p_low_ind, long f_inv_ind,
						  double *_ln_k0_on, double *_kA_on, double *_p_arr,
						  double *_f_inv_arr, long N_p_new, long N_f_new);

		long N_active_sites{0};
		long N_vox{0};
		double D_Hbig{1}, D_H{1};   // big - per-histone chromatin length, normal - real blob diameter
		long *indpos_arr{nullptr};   // N_sites
		long *NHhigh_arr{nullptr};   // N_sites
		long *p_high_ind_arr{nullptr};   // N_sites
		long *p_low_ind_arr{nullptr};   // N_sites
		long *f_inv_ind_arr{nullptr};   // N_sites

		double *ln_k0_on_table{nullptr};   // N_f * N_p
		double *kA_on_table{nullptr};   // N_f * N_p
		double *p_table{nullptr};   // N_p
		double *f_inv_table{nullptr};   // N_f
		long N_p{1}, N_f{1};
		int alloced_arrs{0};

		void print_params() const;
		[[nodiscard]] int indpos_is_acetylated(long indpos) const;
		void alloc_arrs();
		void free_arrs();

		void generate(const long *N_L, int p_gen_mode, long p_high_ind, long NH_high, long p_low_ind,
					  long f_inv_ind, int to_alloc);
		void sort_by_indpos();

		~Acetylation_state();
	};

//	void print_M(const int *M, long Nt, char prefix=0, char suffix='\n');
//	void print_E(const double *E, long Nt, char prefix=0, char suffix='\n');
//	int print_S(const int *s, int L, char prefix);
//	int E_is_valid(const double *E, const double E1, const double E2, int N, int k=0, char prefix=0);
//	void shift_state(int *state, int L, int dx, int dy);
//	int state_is_valid(const int *s, int L, int k=0, char prefix=0);

	int run_bruteforce_C(long *Nt,
						 Proteins_state & prot_init_state,
						 Acetylation_state & acetylation_state,
						 System_parameters & sys_prms,
						 Run_parameters & run_prms,
						 Thermo_log & thermo_log,
						 Rs_log & rs_log,
						 Rho_log & rho_log);

	int get_equilibrated_state(Proteins_state & prot_init_state,
							   Acetylation_state & acetylation_state,
							   System_parameters & sys_prms,
							   Run_parameters & run_prms,
							   Proteins_state * prot_equild_state);


	int init_rand_C(long my_seed);
//	double comp_E(const int* state, int L, const double *e, const double *mu);
//	int comp_M(const int *s, int L);
//	int generate_state(int *s, int L, int mode, int interface_mode, int verbose);
//	double new_spin_energy(const double *e, const double *mu, const int *s_neibs, int s_new);
//	void get_spin_with_neibs(const int *state, int L, int ix, int iy, int *s_group);
//	int swap_move(const int *state, int L, const double *e, const double *mu, int *ix, int *iy, int *ix_new, int *iy_new,
//				  double *dE, const std::set< int > *swap_positions);
////	int long_swap_move(const int *state, int L, const double *e, const double *mu, int *ix, int *iy, int *ix_new, int *iy_new, double *dE);
//	int long_swap_move(const int *state, long L, const double *e, const double *mu, int *ix, int *iy, int *ix_new, int *iy_new, double *dE);
//	int flip_move(const int *state, int L, const double *e, const double *mu, int *ix, int *iy, int *s_new, double *dE);
//	double swap_mode_dE(const int *state, int L, const double *e, const double *mu, int ix, int iy, int ix_new, int iy_new);
//	double long_swap_mode_dE(const int *state, int L, const double *e, const double *mu, int ix, int iy, int ix_new, int iy_new);
//	double short_swap_mode_dE(const int *state, int L, const double *e, const double *mu, int ix, int iy, int ix_new, int iy_new);
//	double flip_mode_dE(const int *state, int L, const double *e, const double *mu, int ix, int iy, int s_new);
//	void set_OP_default(int L2);
//	int get_OP_from_spinsup(int N_spins_up, int L2, int interface_mode);
//
//	bool is_potential_swap_position(int *state, int L, int ix, int iy);
//	void update_neib_potpos(int *state, int L, int ix, int iy, std::set< int > *positions);
//	void find_potential_swaps(int *state, int L, std::set< int > *positions);
//
//
//	void cluster_state_C(const int *s, int L, int *cluster_element_inds, int *cluster_sizes, int *cluster_types, int *N_clusters, int *is_checked);
//	int add_to_cluster(const int* s, int L, int* is_checked, int* cluster, int* cluster_size, int pos, int cluster_label, int cluster_specie);
////	int is_infinite_cluster(const int* cluster, const int* cluster_size, int L, char *present_rows, char *present_columns);
//	void uncheck_state(int *is_checked, int N);
//	void clear_cluster(int* cluster, int *cluster_size);
//	void clear_clusters(int *clusters, int *cluster_sizes, int *N_clusters);
//	int get_max_CS_C(int *state, int L);
}

py::tuple run_bruteforce(py_c_arr(double) rho_init_state, py_c_arr(double) Rs_init, py_c_arr(long) p_indpos_arr,
						 py_c_arr(int) droplet_on_inds_init,
						 double dt, long Nt_max,
						 long thermo_stride, long Rs_stride, long rho_stride,
						 py_c_arr(long) N_L_new, double l_voxel,
						 double ln_k0_off, double kA_off,
						 py_c_arr(double) ln_k0_on, py_c_arr(double) kA_on,
						 double R_0, double rho_coex, double rho_l, double R_S0, double S0, double D_coef,
						 py_c_arr(long) p_high_ind_arr, py_c_arr(long) NHhigh_arr,
						 py_c_arr(long) p_low_ind_arr, py_c_arr(long) f_inv_ind_arr,
						 py_c_arr(double) p_table, py_c_arr(double) f_inv_table,
						 double D_Hbig, double D_H, double S_thr, double ktot_thr,
						 int equil_mode, int to_precomp_dist, int R_crit_mode, int PBC_mode, int accuracy_mode,
						 std::optional<int> verbose_optional);
//void print_state(py_c_arr(int) state);
py::int_ init_rand(long my_seed);
py::int_ set_verbose(int new_verbose);
py::int_ get_verbose();
py::int_ get_seed();
//void print_possible_move_modes();
py::tuple get_equil_modes();
py::tuple get_accuracy_modes();
py::tuple get_Rcrit_modes();
py::tuple get_p_gen_modes();
py::tuple get_rho_gen_modes();

template <typename T> T cmpfunc_inrc (const void * a, const void * b) {	return ( *(T*)a - *(T*)b ); }
template <typename T> T cmpfunc_decr (const void * a, const void * b) {	return -cmpfunc_inrc<T>(a, b); }

#endif //VOXELS_VOXELS_H

