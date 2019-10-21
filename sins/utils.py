
import collections
import datetime

import numpy as np


def to_list(x, length=None):
    """
    Often a list is required, but for convenience it is desired to enter a
    single object, e.g. string.

    Complicated corner cases are e.g. `range()` and `dict.values()`, which are
    handled here.

    >>> to_list(1)
    [1]
    >>> to_list([1])
    [1]
    >>> to_list((i for i in range(3)))
    [0, 1, 2]
    >>> to_list(np.arange(3))
    [0, 1, 2]
    >>> to_list({'a': 1})
    [{'a': 1}]
    >>> to_list({'a': 1}.keys())
    ['a']
    >>> to_list('ab')
    ['ab']
    >>> from pathlib import Path
    >>> to_list(Path('/foo/bar'))
    [PosixPath('/foo/bar')]
    """
    # Important cases (change type):
    #  - generator -> list
    #  - dict_keys -> list
    #  - dict_values -> list
    #  - np.array -> list (discussable)
    # Important cases (list of original object):
    #  - dict -> list of dict

    def to_list_helper(x_):
        return [x_] * (1 if length is None else length)

    if isinstance(x, collections.Mapping):
        x = to_list_helper(x)
    elif isinstance(x, str):
        x = to_list_helper(x)
    elif isinstance(x, collections.Sequence):
        pass
    elif isinstance(x, collections.Iterable):
        x = list(x)
    else:
        x = to_list_helper(x)

    if length is not None:
        assert len(x) == length, (len(x), length)
    return x


def nested_op(
        func,
        arg1, *args,
        broadcast=False,
        handle_dataclass=False,
        keep_type=True,
        mapping_type=collections.Mapping,
        sequence_type=(tuple, list),

):
    """This function is `nested_map` with a fancy name.

    Applies the function "func" to the leafs of the nested data structures.
    This is similar to the map function that applies the function the the
    elements of an iterable input (e.g. list).

    CB: Should handle_dataclass be True or False?
        Other suggestions for the name "handle_dataclass"?

    >>> import operator
    >>> nested_op(operator.add, (3, 5), (7, 11))  # like map
    (10, 16)
    >>> nested_op(operator.add, {'a': (3, 5)}, {'a': (7, 11)})  # with nested
    {'a': (10, 16)}

    >>> nested_op(\
    lambda x, y: x + 3*y, dict(a=[1], b=dict(c=4)), dict(a=[0], b=dict(c=1)))
    {'a': [1], 'b': {'c': 7}}
    >>> arg1, arg2 = dict(a=1, b=dict(c=[1,1])), dict(a=0, b=[1,3])
    >>> nested_op(\
    lambda x, y: x + 3*y, arg1, arg2)
    Traceback (most recent call last):
    ...
    AssertionError: ({'c': [1, 1]}, ([1, 3],))

    Note the broadcasting behavior (arg2.b is broadcasted to arg2.b.c)
    >>> nested_op(\
    lambda x, y: x + 3*y, arg1, arg2, broadcast=True)
    {'a': 1, 'b': {'c': [4, 10]}}

    >>> import dataclasses
    >>> @dataclasses.dataclass
    ... class Data:
    ...     a: int
    ...     b: int
    >>> nested_op(operator.add, Data(3, 5), Data(7, 11), handle_dataclass=True)
    Data(a=10, b=16)

    Args:
        func:
        arg1:
        *args:
        broadcast:
        handle_dataclass: Treat dataclasses as "nested" type or not
        keep_type: Keep the types in the nested structure of arg1 for the
            output or use dict and list as types for the output.
        mapping_type: Types that are interpreted as mapping.
        sequence_type: Types that are interpreted as sequence.

    Returns:

    """
    if isinstance(arg1, mapping_type):
        if not broadcast:
            assert all(
                [isinstance(arg, mapping_type) and arg.keys() == arg1.keys()
                 for arg in args]), (arg1, args)
        else:
            assert all(
                [not isinstance(arg, mapping_type) or arg.keys() == arg1.keys()
                 for arg in args]), (arg1, args)
        keys = arg1.keys()
        output = {
            key: nested_op(
                func,
                arg1[key],
                *[arg[key] if isinstance(arg, mapping_type) else arg
                  for arg in args],
                broadcast=broadcast,
                mapping_type=mapping_type,
                sequence_type=sequence_type,
            )
            for key in keys
        }
        if keep_type:
            output = arg1.__class__(output)
        return output
    elif isinstance(arg1, sequence_type):
        if not broadcast:
            assert all([
                isinstance(arg, sequence_type) and len(arg) == len(arg1)
                for arg in args
            ]), (arg1, args)
        else:
            assert all([
                not isinstance(arg, sequence_type) or len(arg) == len(arg1)
                for arg in args
            ]), (arg1, args)
        output = [
            nested_op(
                func,
                arg1[j],
                *[
                    arg[j] if isinstance(arg, sequence_type) else arg
                    for arg in args
                ],
                broadcast=broadcast,
                mapping_type=mapping_type,
                sequence_type=sequence_type,
            )
            for j in range(len(arg1))
        ]
        if keep_type:
            output = arg1.__class__(output)
        return output
    elif handle_dataclass and hasattr(arg1, '__dataclass_fields__'):
        if not broadcast:
            assert all([
                hasattr(arg, '__dataclass_fields__')
                and arg.__dataclass_fields__ == arg1.__dataclass_fields__
                for arg in args
            ]), (arg1, args)
        else:
            assert all([
                not hasattr(arg, '__dataclass_fields__')
                or arg.__dataclass_fields__ == arg1.__dataclass_fields__
                for arg in args
            ]), (arg1, args)
        return arg1.__class__(
            **{
                f_key: nested_op(
                    func,
                    getattr(arg1, f_key),
                    *[getattr(arg, f_key)
                      if hasattr(arg, '__dataclass_fields__')
                      else arg
                      for arg in args
                    ],
                    broadcast=broadcast,
                    mapping_type=mapping_type,
                    sequence_type=sequence_type,
                )
                for f_key in arg1.__dataclass_fields__
            }
        )

    return func(arg1, *args)


def timestamp(fmt='%Y-%m-%d-%H-%M-%S'):
    return datetime.datetime.now().strftime(fmt)
