package cg

import (
	"fmt"
	"io"
	"math"
)

// Symmetric linear map from R^n to R^n.
type Func func(x []float64) []float64

// Minimizes 1/2 x^T A x - b^T x by solving A x = b.
// A must be symmetric and positive definite.
func Solve(a Func, b, x0 []float64, tol float64, iter int, debug io.Writer) ([]float64, error) {
	seq := NewSeq(a, b, x0)
	var (
		prevObj = math.Inf(1) // Check that objective is monotonic decreasing.
		minRes  = math.Inf(1) // Remember minimum residual thus far.
		x       []float64
	)
	for i := 0; !seq.Final(); i++ {
		if iter > 0 && !(i < iter) {
			break
		}
		if err := seq.Iter(); err != nil {
			return nil, err
		}
		res := seq.Residual()
		if res <= minRes {
			x = seq.Solution()
		}
		if debug != nil {
			obj := seq.Objective()
			var warn string
			if obj > prevObj {
				warn = "incr"
			}
			prevObj = obj
			fmt.Fprintf(debug, "%6d: f:%13.6e r:%10.3e %4s\n", i, obj, res, warn)
		}
		if res <= tol {
			break
		}
	}
	return x, nil
}

type Seq struct {
	a     Func
	b     []float64
	state state
}

type state struct {
	x, r, p []float64
}

func (curr state) Next(a Func, b []float64) (state, error) {
	var next state
	// Compute alpha (line search) and take step.
	ap := a(curr.p)
	pap := dot(curr.p, ap)
	if pap == 0 {
		// Can't divide by zero.
		return state{}, fmt.Errorf("dot(p, A*p) = 0")
	}
	alpha := sqrnorm(curr.r) / pap
	next.x = plusScaled(curr.x, alpha, curr.p)
	next.r = plusScaled(curr.r, -alpha, ap) // minus(b, a(next.x))
	// Compute beta and take step.
	beta := sqrnorm(next.r) / sqrnorm(curr.r)
	next.p = plusScaled(next.r, beta, curr.p)
	return next, nil
}

func NewSeq(a Func, b, x []float64) *Seq {
	var init state
	init.x = clone(x)
	// Initial direction is that of steepest descent.
	ax := a(init.x)
	init.r = minus(b, ax)
	init.p = clone(init.r)
	return &Seq{a, b, init}
}

func (seq *Seq) Final() bool {
	r := seq.state.r
	return sqrnorm(r) == 0
}

// Performs one iteration.
func (seq *Seq) Iter() error {
	next, err := seq.state.Next(seq.a, seq.b)
	if err != nil {
		return err
	}
	seq.state = next
	return nil
}

func (seq *Seq) Solution() []float64 {
	return seq.state.x
}

// Returns 1/2 x' A x - b' x.
func (seq *Seq) Objective() float64 {
	// r = Ax - b
	// 1/2 x'r = 1/2 x'Ax - 1/2 x'b
	// 1/2 x'(r-b) = 1/2 x'Ax - x'b
	x, r := seq.state.x, seq.state.r
	return (dot(x, r) - dot(x, seq.b)) / 2
}

// Returns ||Ax - b|| / ||b||.
func (seq *Seq) Residual() float64 {
	return math.Sqrt(sqrnorm(seq.state.r) / sqrnorm(seq.b))
}
