#ifndef _HUNGARIAN_ALG_H_
#define _HUNGARIAN_ALG_H_

#include <vector>
#include <iostream>
#include <limits>
#include <time.h>
#include <assert.h>

// http://community.topcoder.com/tc?module=Static&d1=tutorials&d2=hungarianAlgorithm
typedef float track_t;
typedef std::vector<int> assignments_t;
typedef std::vector<track_t> dist_matrix_t;

class AssignmentProblemSolver
{
private:
	// --------------------------------------------------------------------------
	// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
	// --------------------------------------------------------------------------
	void assignmentoptimal(assignments_t& assignment, track_t& cost, const dist_matrix_t& dist_matrix_in, size_t n_of_rows, size_t n_of_columns);
	void buildassignmentvector(assignments_t& assignment, bool *star_matrix, size_t n_of_rows, size_t n_of_columns);
	void computeassignmentcost(const assignments_t& assignment, track_t& cost, const dist_matrix_t& dist_matrix_in, size_t n_of_rows);
	void step2a(assignments_t& assignment, track_t *dist_matrix, bool *star_matrix, bool *new_star_matrix, bool *prime_matrix, bool *covered_columns, bool *covered_rows, size_t n_of_rows, size_t n_of_columns, size_t minDim);
	void step2b(assignments_t& assignment, track_t *dist_matrix, bool *star_matrix, bool *new_star_matrix, bool *prime_matrix, bool *covered_columns, bool *covered_rows, size_t n_of_rows, size_t n_of_columns, size_t minDim);
	void step3(assignments_t& assignment, track_t *dist_matrix, bool *star_matrix, bool *new_star_matrix, bool *prime_matrix, bool *covered_columns, bool *covered_rows, size_t n_of_rows, size_t n_of_columns, size_t minDim);
	void step4(assignments_t& assignment, track_t *dist_matrix, bool *star_matrix, bool *new_star_matrix, bool *prime_matrix, bool *covered_columns, bool *covered_rows, size_t n_of_rows, size_t n_of_columns, size_t minDim, size_t row, size_t col);
	void step5(assignments_t& assignment, track_t *dist_matrix, bool *star_matrix, bool *new_star_matrix, bool *prime_matrix, bool *covered_columns, bool *covered_rows, size_t n_of_rows, size_t n_of_columns, size_t minDim);
	// --------------------------------------------------------------------------
	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	// --------------------------------------------------------------------------
	void assignmentsuboptimal1(assignments_t& assignment, track_t& cost, const dist_matrix_t& dist_matrix_in, size_t n_of_rows, size_t n_of_columns);
	// --------------------------------------------------------------------------
	// Computes a suboptimal solution. Good for cases with many forbidden assignments.
	// --------------------------------------------------------------------------
	void assignmentsuboptimal2(assignments_t& assignment, track_t& cost, const dist_matrix_t& dist_matrix_in, size_t n_of_rows, size_t n_of_columns);

public:
	enum TMethod
	{
		optimal,
		many_forbidden_assignments,
		without_forbidden_assignments
	};

	AssignmentProblemSolver();
	~AssignmentProblemSolver();
	track_t Solve(const dist_matrix_t& dist_matrix_in, size_t n_of_rows, size_t n_of_columns, assignments_t& assignment, TMethod Method = optimal);
};

#endif