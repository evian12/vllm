// SPDX-License-Identifier: Apache-2.0
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>

static PyObject *encode(PyObject *module, PyObject *tokens) {
    (void) module;
    int is_list = PyList_CheckExact(tokens);
    if (!is_list && !PyTuple_CheckExact(tokens)) {
        Py_RETURN_NONE;
    }

    Py_ssize_t size = is_list ? PyList_GET_SIZE(tokens) : PyTuple_GET_SIZE(tokens);
    /* Keep the fast path's temporary storage bounded. Long inputs still use
       the original streaming code. */
    if (size > 32768) {
        Py_RETURN_NONE;
    }

    PyObject *bytes = PyBytes_FromStringAndSize(NULL, size * 4 + 8);
    if (bytes == NULL) {
        return NULL;
    }

    unsigned char *dest = (unsigned char *) PyBytes_AS_STRING(bytes);
    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = is_list ? PyList_GET_ITEM(tokens, i) : PyTuple_GET_ITEM(tokens, i);
        if (!PyLong_CheckExact(item)) {
            Py_DECREF(bytes);
            Py_RETURN_NONE;
        }
        unsigned long value = PyLong_AsUnsignedLong(item);
        if (PyErr_Occurred() || value > UINT32_MAX) {
            PyErr_Clear();
            Py_DECREF(bytes);
            Py_RETURN_NONE;
        }
        for (unsigned int lane = 0; lane < 4; ++lane) {
            *dest++ = (unsigned char) (value >> (lane * 8));
        }
    }

    uint64_t count = (uint64_t) size;
    for (unsigned int lane = 0; lane < 8; ++lane) {
        *dest++ = (unsigned char) (count >> (lane * 8));
    }
    return bytes;
}

static PyMethodDef methods[] = {
    {"encode", encode, METH_O, "Encode exact int lists or tuples as uint32 LE tokens and a uint64 LE length."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef extension = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "_cdb_av_prefix_pack",
    .m_size = -1,
    .m_methods = methods,
};

PyMODINIT_FUNC PyInit__cdb_av_prefix_pack(void) {
    return PyModule_Create(&extension);
}
