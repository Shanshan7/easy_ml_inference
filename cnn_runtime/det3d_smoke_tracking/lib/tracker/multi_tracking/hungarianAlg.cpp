#include "hungarianAlg.h"
#include <limits>

AssignmentProblemSolver::AssignmentProblemSolver()
{
}

AssignmentProblemSolver::~AssignmentProblemSolver()
{
}

track_t AssignmentProblemSolver::Solve(
	const dist_matrix_t& dist_matrix_in,
	size_t n_of_rows,
	size_t n_of_columns,
	std::vector<int>& assignment,
	TMethod Method
	)
{
	assignment.resize(n_of_rows, -1);

	track_t cost = 0;

	switch (Method)
	{
	case optimal:
		assignmentoptimal(assignment, cost, dist_matrix_in, n_of_rows, n_of_columns);
		break;

	case many_forbidden_assignments:
		assignmentsuboptimal1(assignment, cost, dist_matrix_in, n_of_rows, n_of_columns);
		break;

	case without_forbidden_assignments:
		assignmentsuboptimal2(assignment, cost, dist_matrix_in, n_of_rows, n_of_columns);
		break;
	}

	return cost;
}
// --------------------------------------------------------------------------
// Computes the optimal assignment (minimum overall costs) using Munkres algorithm.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentoptimal(assignments_t& assignment, track_t& cost, const dist_matrix_t& dist_matrix_in, size_t n_of_rows, size_t n_of_columns)
{
	// Generate distance cv::Matrix 
	// and check cv::Matrix elements positiveness :)

	// Total elements number
	size_t n_of_elements = n_of_rows * n_of_columns;
	// Memory allocation
	track_t* dist_matrix = (track_t *)malloc(n_of_elements * sizeof(track_t));
	// Pointer to last element
	track_t* dist_matrix_end = dist_matrix + n_of_elements;

	for (size_t row = 0; row < n_of_elements; row++)
	{
		track_t value = dist_matrix_in[row];
		assert(value >= 0);
		dist_matrix[row] = value;
	}

	// Memory allocation
	bool* covered_columns = (bool*)calloc(n_of_columns, sizeof(bool));
	bool* covered_rows = (bool*)calloc(n_of_rows, sizeof(bool));
	bool* star_matrix = (bool*)calloc(n_of_elements, sizeof(bool));
	bool* prime_matrix = (bool*)calloc(n_of_elements, sizeof(bool));
	bool* new_star_matrix = (bool*)calloc(n_of_elements, sizeof(bool)); /* used in step4 */

	/* preliminary steps */
	if (n_of_rows <= n_of_columns)
	{
		for (size_t row = 0; row < n_of_rows; row++)
		{
			/* find the smallest element in the row */
			track_t* dist_matrix_temp = dist_matrix + row;
			track_t  minValue = *dist_matrix_temp;
			dist_matrix_temp += n_of_rows;
			while (dist_matrix_temp < dist_matrix_end)
			{
				track_t value = *dist_matrix_temp;
				if (value < minValue)
				{
					minValue = value;
				}
				dist_matrix_temp += n_of_rows;
			}
			/* subtract the smallest element from each element of the row */
			dist_matrix_temp = dist_matrix + row;
			while (dist_matrix_temp < dist_matrix_end)
			{
				*dist_matrix_temp -= minValue;
				dist_matrix_temp += n_of_rows;
			}
		}
		/* Steps 1 and 2a */
		for (size_t row = 0; row < n_of_rows; row++)
		{
			for (size_t col = 0; col < n_of_columns; col++)
			{
				if (dist_matrix[row + n_of_rows*col] == 0)
				{
					if (!covered_columns[col])
					{
						star_matrix[row + n_of_rows * col] = true;
						covered_columns[col] = true;
						break;
					}
				}
			}
		}
	}
	else /* if(n_of_rows > n_of_columns) */
	{
		for (size_t col = 0; col < n_of_columns; col++)
		{
			/* find the smallest element in the column */
			track_t* dist_matrix_temp = dist_matrix + n_of_rows*col;
			track_t* columnEnd = dist_matrix_temp + n_of_rows;
			track_t  minValue = *dist_matrix_temp++;
			while (dist_matrix_temp < columnEnd)
			{
				track_t value = *dist_matrix_temp++;
				if (value < minValue)
				{
					minValue = value;
				}
			}
			/* subtract the smallest element from each element of the column */
			dist_matrix_temp = dist_matrix + n_of_rows*col;
			while (dist_matrix_temp < columnEnd)
			{
				*dist_matrix_temp++ -= minValue;
			}
		}
		/* Steps 1 and 2a */
		for (size_t col = 0; col < n_of_columns; col++)
		{
			for (size_t row = 0; row < n_of_rows; row++)
			{
				if (dist_matrix[row + n_of_rows*col] == 0)
				{
					if (!covered_rows[row])
					{
						star_matrix[row + n_of_rows*col] = true;
						covered_columns[col] = true;
						covered_rows[row] = true;
						break;
					}
				}
			}
		}

		for (size_t row = 0; row < n_of_rows; row++)
		{
			covered_rows[row] = false;
		}
	}
	/* move to step 2b */
	step2b(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows, n_of_rows, n_of_columns, (n_of_rows <= n_of_columns) ? n_of_rows : n_of_columns);
	/* compute cost and remove invalid assignments */
	computeassignmentcost(assignment, cost, dist_matrix_in, n_of_rows);
	/* free allocated memory */
	free(dist_matrix);
	free(covered_columns);
	free(covered_rows);
	free(star_matrix);
	free(prime_matrix);
	free(new_star_matrix);
	return;
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::buildassignmentvector(assignments_t& assignment, bool *star_matrix, size_t n_of_rows, size_t n_of_columns)
{
    for (size_t row = 0; row < n_of_rows; row++)
	{
        for (size_t col = 0; col < n_of_columns; col++)
		{
			if (star_matrix[row + n_of_rows * col])
			{
				assignment[row] = static_cast<int>(col);
				break;
			}
		}
	}
}
// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::computeassignmentcost(const assignments_t& assignment, track_t& cost, const dist_matrix_t& dist_matrix_in, size_t n_of_rows)
{
	for (size_t row = 0; row < n_of_rows; row++)
	{
		const int col = assignment[row];
		if (col >= 0)
		{
			cost += dist_matrix_in[row + n_of_rows * col];
		}
	}
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2a(assignments_t& assignment, track_t *dist_matrix, bool *star_matrix, bool *new_star_matrix, bool *prime_matrix, bool *covered_columns, bool *covered_rows, size_t n_of_rows, size_t n_of_columns, size_t minDim)
{
	bool *star_matrix_temp, *columnEnd;
	/* cover every column containing a starred zero */
	for (size_t col = 0; col < n_of_columns; col++)
	{
		star_matrix_temp = star_matrix + n_of_rows * col;
		columnEnd = star_matrix_temp + n_of_rows;
		while (star_matrix_temp < columnEnd)
		{
			if (*star_matrix_temp++)
			{
				covered_columns[col] = true;
				break;
			}
		}
	}
	/* move to step 3 */
	step2b(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows, n_of_rows, n_of_columns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step2b(assignments_t& assignment, track_t *dist_matrix, bool *star_matrix, bool *new_star_matrix, bool *prime_matrix, bool *covered_columns, bool *covered_rows, size_t n_of_rows, size_t n_of_columns, size_t minDim)
{
	/* count covered columns */
    size_t nOfCoveredColumns = 0;
    for (size_t col = 0; col < n_of_columns; col++)
	{
		if (covered_columns[col])
		{
			nOfCoveredColumns++;
		}
	}
	if (nOfCoveredColumns == minDim)
	{
		/* algorithm finished */
		buildassignmentvector(assignment, star_matrix, n_of_rows, n_of_columns);
	}
	else
	{
		/* move to step 3 */
		step3(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows, n_of_rows, n_of_columns, minDim);
	}
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step3(assignments_t& assignment, track_t *dist_matrix, bool *star_matrix, bool *new_star_matrix, bool *prime_matrix, bool *covered_columns, bool *covered_rows, size_t n_of_rows, size_t n_of_columns, size_t minDim)
{
	bool zerosFound = true;
	while (zerosFound)
	{
		zerosFound = false;
		for (size_t col = 0; col < n_of_columns; col++)
		{
			if (!covered_columns[col])
			{
				for (size_t row = 0; row < n_of_rows; row++)
				{
					if ((!covered_rows[row]) && (dist_matrix[row + n_of_rows*col] == 0))
					{
						/* prime zero */
						prime_matrix[row + n_of_rows*col] = true;
						/* find starred zero in current row */
                        size_t starCol = 0;
						for (; starCol < n_of_columns; starCol++)
						{
							if (star_matrix[row + n_of_rows * starCol])
							{
								break;
							}
						}
						if (starCol == n_of_columns) /* no starred zero found */
						{
							/* move to step 4 */
							step4(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows, n_of_rows, n_of_columns, minDim, row, col);
							return;
						}
						else
						{
							covered_rows[row] = true;
							covered_columns[starCol] = false;
							zerosFound = true;
							break;
						}
					}
				}
			}
		}
	}
	/* move to step 5 */
	step5(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows, n_of_rows, n_of_columns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step4(assignments_t& assignment, track_t *dist_matrix, bool *star_matrix, bool *new_star_matrix, bool *prime_matrix, bool *covered_columns, bool *covered_rows, size_t n_of_rows, size_t n_of_columns, size_t minDim, size_t row, size_t col)
{
	const size_t n_of_elements = n_of_rows * n_of_columns;
	/* generate temporary copy of star_matrix */
	for (size_t n = 0; n < n_of_elements; n++)
	{
		new_star_matrix[n] = star_matrix[n];
	}
	/* star current zero */
	new_star_matrix[row + n_of_rows*col] = true;
	/* find starred zero in current column */
	size_t starCol = col;
	size_t starRow = 0;
	for (; starRow < n_of_rows; starRow++)
	{
		if (star_matrix[starRow + n_of_rows * starCol])
		{
			break;
		}
	}
	while (starRow < n_of_rows)
	{
		/* unstar the starred zero */
		new_star_matrix[starRow + n_of_rows*starCol] = false;
		/* find primed zero in current row */
		size_t primeRow = starRow;
		size_t primeCol = 0;
		for (; primeCol < n_of_columns; primeCol++)
		{
			if (prime_matrix[primeRow + n_of_rows * primeCol])
			{
				break;
			}
		}
		/* star the primed zero */
		new_star_matrix[primeRow + n_of_rows*primeCol] = true;
		/* find starred zero in current column */
		starCol = primeCol;
		for (starRow = 0; starRow < n_of_rows; starRow++)
		{
			if (star_matrix[starRow + n_of_rows * starCol])
			{
				break;
			}
		}
	}
	/* use temporary copy as new star_matrix */
	/* delete all primes, uncover all rows */
    for (size_t n = 0; n < n_of_elements; n++)
	{
		prime_matrix[n] = false;
		star_matrix[n] = new_star_matrix[n];
	}
    for (size_t n = 0; n < n_of_rows; n++)
	{
		covered_rows[n] = false;
	}
	/* move to step 2a */
	step2a(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows, n_of_rows, n_of_columns, minDim);
}

// --------------------------------------------------------------------------
//
// --------------------------------------------------------------------------
void AssignmentProblemSolver::step5(assignments_t& assignment, track_t *dist_matrix, bool *star_matrix, bool *new_star_matrix, bool *prime_matrix, bool *covered_columns, bool *covered_rows, size_t n_of_rows, size_t n_of_columns, size_t minDim)
{
	/* find smallest uncovered element h */
	float h = std::numeric_limits<track_t>::max();
	for (size_t row = 0; row < n_of_rows; row++)
	{
		if (!covered_rows[row])
		{
			for (size_t col = 0; col < n_of_columns; col++)
			{
				if (!covered_columns[col])
				{
					const float value = dist_matrix[row + n_of_rows*col];
					if (value < h)
					{
						h = value;
					}
				}
			}
		}
	}
	/* add h to each covered row */
	for (size_t row = 0; row < n_of_rows; row++)
	{
		if (covered_rows[row])
		{
			for (size_t col = 0; col < n_of_columns; col++)
			{
				dist_matrix[row + n_of_rows*col] += h;
			}
		}
	}
	/* subtract h from each uncovered column */
	for (size_t col = 0; col < n_of_columns; col++)
	{
		if (!covered_columns[col])
		{
			for (size_t row = 0; row < n_of_rows; row++)
			{
				dist_matrix[row + n_of_rows*col] -= h;
			}
		}
	}
	/* move to step 3 */
	step3(assignment, dist_matrix, star_matrix, new_star_matrix, prime_matrix, covered_columns, covered_rows, n_of_rows, n_of_columns, minDim);
}


// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases without forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal2(assignments_t& assignment, track_t& cost, const dist_matrix_t& dist_matrix_in, size_t n_of_rows, size_t n_of_columns)
{
	/* make working copy of distance Matrix */
	const size_t n_of_elements = n_of_rows * n_of_columns;
	float* dist_matrix = (float*)malloc(n_of_elements * sizeof(float));
	for (size_t n = 0; n < n_of_elements; n++)
	{
		dist_matrix[n] = dist_matrix_in[n];
	}

	/* recursively search for the minimum element and do the assignment */
	for (;;)
	{
		/* find minimum distance observation-to-track pair */
		float minValue = std::numeric_limits<track_t>::max();
		size_t tmpRow = 0;
		size_t tmpCol = 0;
		for (size_t row = 0; row < n_of_rows; row++)
		{
			for (size_t col = 0; col < n_of_columns; col++)
			{
				const float value = dist_matrix[row + n_of_rows*col];
				if (value != std::numeric_limits<track_t>::max() && (value < minValue))
				{
					minValue = value;
					tmpRow = row;
					tmpCol = col;
				}
			}
		}

		if (minValue != std::numeric_limits<track_t>::max())
		{
			assignment[tmpRow] = static_cast<int>(tmpCol);
			cost += minValue;
			for (size_t n = 0; n < n_of_rows; n++)
			{
				dist_matrix[n + n_of_rows*tmpCol] = std::numeric_limits<track_t>::max();
			}
			for (size_t n = 0; n < n_of_columns; n++)
			{
				dist_matrix[tmpRow + n_of_rows*n] = std::numeric_limits<track_t>::max();
			}
		}
		else
		{
			break;
		}
	}

	free(dist_matrix);
}
// --------------------------------------------------------------------------
// Computes a suboptimal solution. Good for cases with many forbidden assignments.
// --------------------------------------------------------------------------
void AssignmentProblemSolver::assignmentsuboptimal1(assignments_t& assignment, track_t& cost, const dist_matrix_t& dist_matrix_in, size_t n_of_rows, size_t n_of_columns)
{
	/* make working copy of distance Matrix */
	const size_t n_of_elements = n_of_rows * n_of_columns;
	float* dist_matrix = (float *)malloc(n_of_elements * sizeof(float));
    for (size_t n = 0; n < n_of_elements; n++)
	{
		dist_matrix[n] = dist_matrix_in[n];
	}

	/* allocate memory */
	int* nOfValidObservations = (int *)calloc(n_of_rows, sizeof(int));
	int* nOfValidTracks = (int *)calloc(n_of_columns, sizeof(int));

	/* compute number of validations */
	bool infiniteValueFound = false;
	bool finiteValueFound = false;
	for (size_t row = 0; row < n_of_rows; row++)
	{
		for (size_t col = 0; col < n_of_columns; col++)
		{
			if (dist_matrix[row + n_of_rows*col] != std::numeric_limits<track_t>::max())
			{
				nOfValidTracks[col] += 1;
				nOfValidObservations[row] += 1;
				finiteValueFound = true;
			}
			else
			{
				infiniteValueFound = true;
			}
		}
	}

	if (infiniteValueFound)
	{
		if (!finiteValueFound)
		{
			return;
		}
		bool repeatSteps = true;

		while (repeatSteps)
		{
			repeatSteps = false;

			/* step 1: reject assignments of multiply validated tracks to singly validated observations		 */
			for (size_t col = 0; col < n_of_columns; col++)
			{
				bool singleValidationFound = false;
				for (size_t row = 0; row < n_of_rows; row++)
				{
					if (dist_matrix[row + n_of_rows * col] != std::numeric_limits<track_t>::max() && (nOfValidObservations[row] == 1))
					{
						singleValidationFound = true;
						break;
					}

					if (singleValidationFound)
					{
						for (size_t row = 0; row < n_of_rows; row++)
							if ((nOfValidObservations[row] > 1) && dist_matrix[row + n_of_rows*col] != std::numeric_limits<track_t>::max())
							{
								dist_matrix[row + n_of_rows*col] = std::numeric_limits<track_t>::max();
								nOfValidObservations[row] -= 1;
								nOfValidTracks[col] -= 1;
								repeatSteps = true;
							}
					}
				}
			}

			/* step 2: reject assignments of multiply validated observations to singly validated tracks */
			if (n_of_columns > 1)
			{
				for (size_t row = 0; row < n_of_rows; row++)
				{
					bool singleValidationFound = false;
                    for (size_t col = 0; col < n_of_columns; col++)
					{
						if (dist_matrix[row + n_of_rows*col] != std::numeric_limits<track_t>::max() && (nOfValidTracks[col] == 1))
						{
							singleValidationFound = true;
							break;
						}
					}

					if (singleValidationFound)
					{
						for (size_t col = 0; col < n_of_columns; col++)
						{
							if ((nOfValidTracks[col] > 1) && dist_matrix[row + n_of_rows*col] != std::numeric_limits<track_t>::max())
							{
								dist_matrix[row + n_of_rows*col] = std::numeric_limits<track_t>::max();
								nOfValidObservations[row] -= 1;
								nOfValidTracks[col] -= 1;
								repeatSteps = true;
							}
						}
					}
				}
			}
		} /* while(repeatSteps) */

		/* for each multiply validated track that validates only with singly validated  */
		/* observations, choose the observation with minimum distance */
        for (size_t row = 0; row < n_of_rows; row++)
		{
			if (nOfValidObservations[row] > 1)
			{
				bool allSinglyValidated = true;
				float minValue = std::numeric_limits<track_t>::max();
				size_t tmpCol = 0;
                for (size_t col = 0; col < n_of_columns; col++)
				{
					const float value = dist_matrix[row + n_of_rows*col];
					if (value != std::numeric_limits<track_t>::max())
					{
						if (nOfValidTracks[col] > 1)
						{
							allSinglyValidated = false;
							break;
						}
						else if ((nOfValidTracks[col] == 1) && (value < minValue))
						{
							tmpCol = col;
							minValue = value;
						}
					}
				}

				if (allSinglyValidated)
				{
					assignment[row] = static_cast<int>(tmpCol);
					cost += minValue;
					for (size_t n = 0; n < n_of_rows; n++)
					{
						dist_matrix[n + n_of_rows*tmpCol] = std::numeric_limits<track_t>::max();
					}
					for (size_t n = 0; n < n_of_columns; n++)
					{
						dist_matrix[row + n_of_rows*n] = std::numeric_limits<track_t>::max();
					}
				}
			}
		}

		// for each multiply validated observation that validates only with singly validated  track, choose the track with minimum distance
        for (size_t col = 0; col < n_of_columns; col++)
		{
			if (nOfValidTracks[col] > 1)
			{
				bool allSinglyValidated = true;
				float minValue = std::numeric_limits<track_t>::max();
				size_t tmpRow = 0;
				for (size_t row = 0; row < n_of_rows; row++)
				{
					const float value = dist_matrix[row + n_of_rows*col];
					if (value != std::numeric_limits<track_t>::max())
					{
						if (nOfValidObservations[row] > 1)
						{
							allSinglyValidated = false;
							break;
						}
						else if ((nOfValidObservations[row] == 1) && (value < minValue))
						{
							tmpRow = row;
							minValue = value;
						}
					}
				}

				if (allSinglyValidated)
				{
					assignment[tmpRow] = static_cast<int>(col);
					cost += minValue;
					for (size_t n = 0; n < n_of_rows; n++)
					{
						dist_matrix[n + n_of_rows*col] = std::numeric_limits<track_t>::max();
					}
					for (size_t n = 0; n < n_of_columns; n++)
					{
						dist_matrix[tmpRow + n_of_rows*n] = std::numeric_limits<track_t>::max();
					}
				}
			}
		}
	} /* if(infiniteValueFound) */


	/* now, recursively search for the minimum element and do the assignment */
	for (;;)
	{
		/* find minimum distance observation-to-track pair */
		float minValue = std::numeric_limits<track_t>::max();
		size_t tmpRow = 0;
		size_t tmpCol = 0;
		for (size_t row = 0; row < n_of_rows; row++)
		{
			for (size_t col = 0; col < n_of_columns; col++)
			{
				const float value = dist_matrix[row + n_of_rows*col];
				if (value != std::numeric_limits<track_t>::max() && (value < minValue))
				{
					minValue = value;
					tmpRow = row;
					tmpCol = col;
				}
			}
		}

		if (minValue != std::numeric_limits<track_t>::max())
		{
			assignment[tmpRow] = static_cast<int>(tmpCol);
			cost += minValue;
			for (size_t n = 0; n < n_of_rows; n++)
			{
				dist_matrix[n + n_of_rows*tmpCol] = std::numeric_limits<track_t>::max();
			}
			for (size_t n = 0; n < n_of_columns; n++)
			{
				dist_matrix[tmpRow + n_of_rows*n] = std::numeric_limits<track_t>::max();
			}
		}
		else
		{
			break;
		}
	}

	/* free allocated memory */
	free(nOfValidObservations);
	free(nOfValidTracks);
	free(dist_matrix);
}
