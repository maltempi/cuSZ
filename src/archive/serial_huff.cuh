/**
 *  @file Huffman.cuh
 *  @author Sheng Di
 *  Modified by Jiannan Tian
 *  @date Jan. 7, 2020
 *  Created on Aug., 2016
 *  @brief Customized Huffman Encoding, Compression and Decompression functions.
 *         Also modified for GPU prototyping (header).
 *  (C) 2016 by Mathematics and Computer Science (MCS), Argonne National Laboratory.
 *      See COPYRIGHT in top-level directory.
 */

#ifndef HUFFMAN_CUH
#define HUFFMAN_CUH

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifdef __CUDACC__

#define KERNEL __global__
#define SUBROUTINE __host__ __device__
#define INLINE __forceinline__
#define ON_DEVICE_FALLBACK2HOST __device__

#else

#define KERNEL
#define SUBROUTINE
#define INLINE inline
#define ON_DEVICE_FALLBACK2HOST

#endif

struct alignas(8) node_t {
    struct node_t *left, *right;
    size_t         freq;
    char           t;  // in_node:0; otherwise:1
    uint32_t       c;
};

typedef struct node_t* node_list;

typedef struct alignas(8) HuffmanTree {
    uint32_t       state_num;
    uint32_t       all_nodes;
    struct node_t* pool;
    node_list *    qqq, *qq;  // the root node of the HuffmanTree is qq[1]
    int            n_nodes;   // n_nodes is for compression
    int            qend;
    uint64_t**     code;
    uint8_t*       cout;
    int            n_inode;  // n_inode is for decompression
} HuffmanTree;

SUBROUTINE node_list new_node(HuffmanTree* huffman_tree, size_t freq, uint32_t c, node_list a, node_list b);
SUBROUTINE void      qinsert(HuffmanTree* ht, node_list n);
SUBROUTINE node_list qremove(HuffmanTree* ht);
SUBROUTINE void      build_code(HuffmanTree* ht, node_list n, int len, uint64_t out1, uint64_t out2);

// auxiliary functions done
SUBROUTINE HuffmanTree* create_huffman_tree(int state_num);

SUBROUTINE node_list new_node(HuffmanTree* huffman_tree, size_t freq, uint32_t c, node_list a, node_list b);

/* priority queue */
SUBROUTINE void qinsert(HuffmanTree* ht, node_list n);

SUBROUTINE node_list qremove(HuffmanTree* ht);

SUBROUTINE void build_code(HuffmanTree* ht, node_list n, int len, uint64_t out1, uint64_t out2);

////////////////////////////////////////////////////////////////////////////////
// internal functions
////////////////////////////////////////////////////////////////////////////////

const int MAX_DEPTH = 32;
//#define MAX_DEPTH 32

typedef struct alignas(8) Stack {
    node_list _a[MAX_DEPTH];
    uint64_t  saved_path[MAX_DEPTH];
    uint64_t  saved_length[MAX_DEPTH];
    uint64_t  depth = 0;
} internal_stack_t;

SUBROUTINE INLINE bool is_empty_tree(internal_stack_t* s);

SUBROUTINE INLINE node_list top(internal_stack_t* s);

template <typename T>
SUBROUTINE INLINE void push_v2(internal_stack_t* s, node_list n, T path, T len);

// TODO check with typing
template <typename T>
SUBROUTINE INLINE node_list pop_v2(internal_stack_t* s, T* path_to_restore, T* length_to_restore);

template <typename Q>
SUBROUTINE void inorder_traverse_v2(HuffmanTree* ht, Q* codebook);

////////////////////////////////////////////////////////////////////////////////
// global functions
////////////////////////////////////////////////////////////////////////////////

ON_DEVICE_FALLBACK2HOST HuffmanTree* global_tree;

template <typename H>
KERNEL void init_huffman_tree_and_get_codebook(int state_num, unsigned int* freq, H* codebook);

#endif
