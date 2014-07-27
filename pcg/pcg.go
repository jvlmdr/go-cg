package pcg

import (
	"fmt"
	"math"

	"github.com/jvlmdr/go-cg/cg"
)

// Minimizes 1/2 x^T A x - b^T x by solving Cinv A x = Cinv b.
// A and Cinv must be symmetric and positive definite.
func Solve(a cg.Func, b []float64, cinv cg.Func, x0 []float64, tol float64, iter int) ([]float64, error) {
	seq := NewSeq(a, b, cinv, x0)
	for i := 0; i < iter && !seq.Final(); i++ {
		if err := seq.Iter(); err != nil {
			return nil, err
		}
		if seq.Residual() <= tol {
			break
		}
	}
	return seq.Solution(), nil
}

type Seq struct {
	a     cg.Func
	b     []float64
	cinv  cg.Func
	state state
}

type state struct {
	x, r, z, p []float64
}

func (curr state) Next(a, cinv cg.Func, b []float64) (state, error) {
	var next state
	// Compute t (line search) and take step.
	ap := a(curr.p)
	pap := dot(curr.p, ap)
	if pap == 0 {
		// Can't divide by zero.
		return state{}, fmt.Errorf("dot(p, A*p) = 0")
	}
	alpha := dot(curr.r, curr.z) / pap
	next.x = plusScaled(curr.x, alpha, curr.p)
	next.r = plusScaled(curr.r, -alpha, ap) // minus(b, a(next.x))
	next.z = cinv(next.r)
	// Compute beta and take step.
	beta := dot(next.z, next.r) / dot(curr.z, curr.r)
	next.p = plusScaled(next.z, beta, curr.p)
	return next, nil
}

func NewSeq(a cg.Func, b []float64, cinv cg.Func, x []float64) *Seq {
	var init state
	// Copy x.
	init.x = clone(x)
	// Initial direction is that of steepest descent.
	ax := a(x)
	init.r = minus(b, ax)
	init.z = cinv(init.r)
	init.p = clone(init.z)
	return &Seq{a, b, cinv, init}
}

func (seq *Seq) Final() bool {
	return sqrnorm(seq.state.r) == 0
}

func (seq *Seq) Iter() error {
	next, err := seq.state.Next(seq.a, seq.cinv, seq.b)
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
