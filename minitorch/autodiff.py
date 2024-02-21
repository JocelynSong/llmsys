from dataclasses import dataclass
from typing import Any, Iterable, List, Tuple

from typing_extensions import Protocol


def central_difference(f: Any, *vals: Any, arg: int = 0, epsilon: float = 1e-6) -> Any:
    r"""
    Computes an approximation to the derivative of `f` with respect to one arg.

    See :doc:`derivative` or https://en.wikipedia.org/wiki/Finite_difference for more details.

    Args:
        f : arbitrary function from n-scalar args to one value
        *vals : n-float values $x_0 \ldots x_{n-1}$
        arg : the number $i$ of the arg to compute the derivative
        epsilon : a small constant

    Returns:
        An approximation of $f'_i(x_0, \ldots, x_{n-1})$
    """
    vals1 = [v for v in vals]
    vals2 = [v for v in vals]
    vals1[arg] = vals1[arg] + epsilon
    vals2[arg] = vals2[arg] - epsilon
    delta = f(*vals1) - f(*vals2)
    return delta / (2 * epsilon)


variable_count = 1


class Variable(Protocol):
    def accumulate_derivative(self, x: Any) -> None:
        pass

    @property
    def unique_id(self) -> int:
        pass

    def is_leaf(self) -> bool:
        pass

    def is_constant(self) -> bool:
        pass

    @property
    def parents(self) -> Iterable["Variable"]:
        pass

    def chain_rule(self, d_output: Any) -> Iterable[Tuple["Variable", Any]]:
        pass


def topological_sort(variable: Variable) -> Iterable[Variable]:
    """
    Computes the topological order of the computation graph.

    Args:
        variable: The right-most variable

    Returns:
        Non-constant Variables in topological order starting from the right.
    """
    q_list = [variable]
    out_degree = {variable.unique_id: 0}
    head = 0
    while head < len(q_list):
        v = q_list[head]
        head += 1
        for p in v.parents:
            if p.unique_id not in out_degree:
                out_degree[p.unique_id] = 0
                if not p.is_constant() and not p.is_leaf():
                    q_list.append(p)
            out_degree[p.unique_id] += 1
    stack = [variable]
    tail = 0
    res = []
    while len(stack) > 0:
        v = stack[-1]
        stack.pop()
        if not v.is_constant():
            res.append(v)
        if v.is_constant() or v.is_leaf():
            continue
        for p in v.parents:
            out_degree[p.unique_id] -= 1
            if out_degree[p.unique_id] == 0:
                stack.append(p)
                tail += 1
    return res


def backpropagate(variable: Variable, deriv: Any) -> None:
    """
    Runs backpropagation on the computation graph in order to
    compute derivatives for the leave nodes.

    Args:
        variable: The right-most variable
        deriv  : Its derivative that we want to propagate backward to the leaves.

    No return. Should write to its results to the derivative values of each leaf through `accumulate_derivative`.
    """
    node_list = topological_sort(variable)
    variable.grad = deriv

    for v in node_list:
        if v.is_leaf():
            continue
        derivatives = v.chain_rule(v.grad)
        for p, d in zip(v.parents, derivatives):
            if p.is_leaf():
                p.accumulate_derivative(d[1])
            else:
                p.grad = d[1]


@dataclass
class Context:
    """
    Context class is used by `Function` to store information during the forward pass.
    """

    no_grad: bool = False
    saved_values: Tuple[Any, ...] = ()

    def save_for_backward(self, *values: Any) -> None:
        "Store the given `values` if they need to be used during backpropagation."
        if self.no_grad:
            return
        self.saved_values = values

    @property
    def saved_tensors(self) -> Tuple[Any, ...]:
        return self.saved_values
