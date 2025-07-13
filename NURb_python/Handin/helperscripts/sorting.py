from numpy import ndarray


def merge(left, right):
    sorted_array = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            sorted_array.append(left[i])
            i += 1
        else:
            sorted_array.append(right[j])
            j += 1

    sorted_array.extend(left[i:])
    sorted_array.extend(right[j:])
    return sorted_array


def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)


def merge_with_indices(left, right, refs):
    sorted_array = []
    i = j = 0

    while i < len(left) and j < len(right):
        if refs[left[i]] < refs[right[j]]:
            sorted_array.append(left[i])
            i += 1
        else:
            sorted_array.append(right[j])
            j += 1

    sorted_array.extend(left[i:])
    sorted_array.extend(right[j:])
    return sorted_array

def merge_sort_indices(indices, refs):
    if len(indices) <= 1:
        return indices
    mid = len(indices) // 2
    left = merge_sort_indices(indices[:mid], refs)
    right = merge_sort_indices(indices[mid:], refs)
    return merge_with_indices(left, right, refs)

