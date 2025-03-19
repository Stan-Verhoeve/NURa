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

def merge_in_place(arr, left, mid, right):
    i, j = left, mid + 1
    while i <= mid and j <= right:
        if arr[i] <= arr[j]:
            i += 1
        else:
            # Element arr[j] is smaller than arr[i], so we need to move arr[j]
            temp = arr[j]
            # Shift all elements between i and j-1 one step to the right
            for k in range(j, i, -1):
                arr[k] = arr[k - 1]
            arr[i] = temp
            i += 1
            mid += 1
            j += 1

def merge_sort_in_place(arr, left=0, right=None):
    if right is None:
        right = len(arr) - 1
    
    if left >= right:
        return
    
    mid = (left + right) // 2
    
    merge_sort_in_place(arr, left, mid)
    merge_sort_in_place(arr, mid + 1, right)
    
    # Merge the two sorted halves in-place
    merge_in_place(arr, left, mid, right)
def is_sorted(arr):
    return all(arr[i] <= arr[i+1] for i in range(len(arr) - 1))
