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
    return  merge(left, right)

def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr) - 1))
