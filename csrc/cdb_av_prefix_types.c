/* SPDX-License-Identifier: Apache-2.0
 * Exact equivalent of all(type(x) is int for x in a builtin list/tuple).
 * No token-range, fingerprint, request, tensor or KV semantics live here.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <limits.h>

static unsigned long long invocation_count = 0;

static PyObject *all_builtin_ints(PyObject *self, PyObject *sequence) {
    (void)self;
    if (invocation_count != ULLONG_MAX) ++invocation_count;
    if (!PyList_CheckExact(sequence) && !PyTuple_CheckExact(sequence)) {
        Py_RETURN_NOTIMPLEMENTED;
    }
    /* CPython holds the GIL; no callback, allocation or token conversion can
       mutate sequence while its borrowed item pointers are examined. */
    Py_ssize_t length = PySequence_Fast_GET_SIZE(sequence);
    PyObject **items = PySequence_Fast_ITEMS(sequence);
    for (Py_ssize_t i = 0; i < length; ++i) {
        if (!PyLong_CheckExact(items[i])) Py_RETURN_FALSE;
    }
    Py_RETURN_TRUE;
}

static PyObject *calls(PyObject *self, PyObject *unused) {
    (void)self;
    (void)unused;
    return PyLong_FromUnsignedLongLong(invocation_count);
}

static PyMethodDef methods[] = {
    {"all_builtin_ints", all_builtin_ints, METH_O,
     "Exact builtin-int type predicate, or NotImplemented for other sequences."},
    {"calls", calls, METH_NOARGS, "Number of actual native predicate invocations."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef module = {
    PyModuleDef_HEAD_INIT, "_cdb_av_prefix_types",
    "Optional CPython type-only predicate; no model or cache state.",
    -1, methods, NULL, NULL, NULL, NULL
};

PyMODINIT_FUNC PyInit__cdb_av_prefix_types(void) {
    return PyModule_Create(&module);
}
