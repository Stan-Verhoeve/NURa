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


def merge_in_place(arr: ndarray, left: int, mid: int, right: int) -> None:
    """
    Helper function for merge_sort

    Parameters
    ----------
    arr : ndarray
        Array to merge in-place
    left : int
        Index of start of left bracket
    mid : int
        Index of mid of array
    right : int
        Index of end of right bracket
    """
    # Inidices to keep track
    i, j = left, mid + 1

    # Loop until we reach mid and right of the bracket
    while i <= mid and j <= right:
        # Elements already sorted
        if arr[i] <= arr[j]:
            i += 1
        else:
            # Move all elements s.t. arr[j] is in front of arr[i]
            temp = arr[j]
            for k in range(j, i, -1):
                arr[k] = arr[k - 1]
            arr[i] = temp

            # Update all indices to reflect moving of arr[j]
            i += 1
            mid += 1
            j += 1


def merge_sort_in_place(arr: ndarray, left: int=0, right: int=None) -> None:
    """
    Merge sort in-place

    Parameters
    ----------
    arr : ndarray
        Array to sort in-place
    left : int
        Index of start of left bracket
    right : int
        Index of end of right bracket
    """
    # Default behaviour
    if right is None:
        right = len(arr) - 1
    
    # Array is already sorted
    if left >= right:
        return
    
    # Midpoint of the current arrai
    mid = (left + right) // 2
    
    # Recursively sub-divide into smaller arrays
    merge_sort_in_place(arr, left, mid)
    merge_sort_in_place(arr, mid + 1, right)

    # Merge the two sorted halves in-place
    merge_in_place(arr, left, mid, right)


def is_sorted(arr):
    return all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
