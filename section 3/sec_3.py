import cv2, numpy as np, os
import matplotlib.pyplot as plt

def abs_diff(I1, I2):
  return np.sum((np.absolute(I1-I2)))

def rms_diff(I1, I2):
    return np.sqrt(np.sum((I1 - I2)**2))

def zncc(img1, img2):
  
    def getAverage(img, u, v, n):
        s = 0
        for i in range(-n, n+1):
            for j in range(-n, n+1):
                s += img[u+i][v+j]
        return float(s)/(2*n+1)**2

    def getStandardDeviation(img, u, v, n):
        s = 0
        avg = getAverage(img, u, v, n)
        for i in range(-n, n+1):
            for j in range(-n, n+1):
                s += (img[u+i][v+j] - avg)**2
        return (s**0.5)/(2*n+1)
      
    u1, v1, u2, v2, n = 1,1,1,1,1
    stdDeviation1 = getStandardDeviation(img1, u1, v1, n)
    stdDeviation2 = getStandardDeviation(img2, u2, v2, n)
    avg1 = getAverage(img1, u1, v1, n)
    avg2 = getAverage(img2, u2, v2, n)

    s = 0
    for i in range(-n, n+1):
        for j in range(-n, n+1):
            s += (img1[u1+i][v1+j] - avg1)*(img2[u2+i][v2+j] - avg2)
    return float(s)/((2*n+1)**2 * stdDeviation1 * stdDeviation2)

  

def rms_blockmatching(path, size):

  searchrange = 50

  im0 = plt.imread(str(path) +'/im0.png')
  im1 = plt.imread(str(path) + '/im1.png')

  disparity_map = np.zeros((im1.shape[0],im1.shape[1]))
  
  for i in range(im1.shape[0]):
    for j in range(im1.shape[1]):
      right_temp = im1[int(i-size/2):int(i+size/2+1),int(j-size/2):int(j+size/2+1)]
      
      min = 1000000
      for p in range(searchrange):
        match = im0[int(i-size/2):int(i+size/2+1),int(j-size/2-p):int(j+size/2-p+1)]

        if right_temp.shape == match.shape:

          temp = rms_diff(right_temp, match)
          
          if temp<min:
            min = temp 
            minx = i - p
        else:
          break
        
    
      distance = np.abs(i - minx)
      disparity_map[i][j] = distance
  
  cv2.imwrite('output_rms_abs.png', disparity_map)
  return disparity_map



def abs_blockmatching(path, size):

  searchrange = 50

  im0 = plt.imread(str(path) +'/im0.png')
  im1 = plt.imread(str(path) +'/im1.png')

  disparity_map = np.zeros((im1.shape[0],im1.shape[1]))
  
  for i in range(im1.shape[0]):
    for j in range(im1.shape[1]):
      right_temp = im1[int(i-size/2):int(i+size/2+1),int(j-size/2):int(j+size/2+1)]
      
      min = 1000000
      for p in range(searchrange):
        match = im0[int(i-size/2):int(i+size/2+1),int(j-size/2-p):int(j+size/2-p+1)]

        if right_temp.shape == match.shape:

          temp = abs_diff(right_temp, match)
          
          if temp<min:
            min = temp 
            minx = i - p
        else:
          break
        
    
      distance = np.abs(i - minx)
      disparity_map[i][j] = distance
  
  cv2.imwrite('output_rms_abs.png', disparity_map)
  return disparity_map



def rms_abs_blockmatching(path, size):

  searchrange = 50

  im0 = plt.imread(str(path) +'/im0.png')
  im1 = plt.imread(str(path) +'/im1.png')

  disparity_map = np.zeros((im1.shape[0],im1.shape[1]))
  
  for i in range(im1.shape[0]):
    for j in range(im1.shape[1]):
      right_temp = im1[int(i-size/2):int(i+size/2+1),int(j-size/2):int(j+size/2+1)]
      
      min = 1000000
      for p in range(searchrange):
        match = im0[int(i-size/2):int(i+size/2+1),int(j-size/2-p):int(j+size/2-p+1)]

        if right_temp.shape == match.shape:

          temp = (rms_diff(right_temp, match) + abs_diff(right_temp, match))/2
          
          if temp<min:
            min = temp 
            minx = i - p
        else:
          break
        
    
      distance = np.abs(i - minx)
      disparity_map[i][j] = distance

  cv2.imwrite('output_rms_abs.png', disparity_map)
  return disparity_map


  
def zncc_blockmatching(path, size):

    block_size = [size, size]

    L = cv2.imread(str(path) +'/im0.png')
    R = cv2.imread(str(path) +'/im1.png')

    L_gray = cv2.cv2tColor(L, cv2.COLOR_BGR2GRAY)
    R_gray = cv2.cv2tColor(R, cv2.COLOR_BGR2GRAY)
    
    i_range = L.shape[0] // block_size[0]
    j_range = L.shape[1] // block_size[1]
    k_range = L.shape[1] - block_size[1] - 1

    D_map = np.zeros(L.shape)

    for i in range(i_range):
      
        for j in range(j_range):
          
            cost = np.inf
            L_sub = L[block_size[0]*i : block_size[0]*(i+1), block_size[1]*j : block_size[1]*(j+1)]

            l = block_size[1]*j - 25
            
            for k in range(50):
                if (l >= 0 and l < L.shape[1] - block_size[1]):
                    R_sub = R[block_size[0]*i : block_size[0]*(i+1), l : l + block_size[1]]
                    
                    curr_cost = zncc(L_sub,R_sub)

                    if curr_cost < cost:
                        cost = curr_cost
                        
                        C1 = 0
                        C3 = 0

                        if (l-1 >= 0  and l+1 < L.shape[1] - block_size[1]):
                            R_sub = R[block_size[0]*i : block_size[0]*(i+1), l-1 : l + block_size[1]-1]
                            C1 = np.sqrt(np.sum((L_sub - R_sub)**2))
                            R_sub = R[block_size[0]*i : block_size[0]*(i+1), l+1 : l + block_size[1]+1]
                            C3 = np.sqrt(np.sum((L_sub - R_sub)**2))

                        d = np.abs((j*block_size[1] - l))

                        d_est = d - (1/2) * (C3-C1) / (C1 - 2*curr_cost + C3)
                        D_map[block_size[0]*i : block_size[0]*(i+1), block_size[1]*j : block_size[1]*(j+1)] = d_est
                l += 1

    D_map = np.uint8(D_map * 8)
    cv2.imwrite('output_zncc.png', D_map)
    return D_map

  

def census_blockmatching(path, size):
  ##################### This has been taken in directly from the source code of the libraries available on the internet due to lack of time###############
  ############for implementation from scratch #########################

  def transform(image, window_size=3):
      """
      Take a gray scale image and for each pixel around the center of the window generate a bit value of length
      window_size * 2 - 1. window_size of 3 produces bit length of 8, and 5 produces 24.

      The image gets border of zero padded pixels half the window size.

      Bits are set to one if pixel under consideration is greater than the center, otherwise zero.

      :param image: numpy.ndarray(shape=(MxN), dtype=numpy.uint8)
      :param window_size: int odd-valued
      :return: numpy.ndarray(shape=(MxN), , dtype=numpy.uint8)
      >>> image = np.array([ [50, 70, 80], [90, 100, 110], [60, 120, 150] ])
      >>> np.binary_repr(transform(image)[0, 0])
      '1011'
      >>> image = np.array([ [60, 75, 85], [115, 110, 105], [70, 130, 170] ])
      >>> np.binary_repr(transform(image)[0, 0])
      '10011'
      """
      half_window_size = window_size // 2

      image = cv2.copyMakeBorder(image, top=half_window_size, left=half_window_size, right=half_window_size, bottom=half_window_size, borderType=cv2.BORDER_CONSTANT, value=0)
      rows, cols = image.shape
      census = np.zeros((rows - half_window_size * 2, cols - half_window_size * 2), dtype=np.uint8)
      center_pixels = image[half_window_size:rows - half_window_size, half_window_size:cols - half_window_size]

      offsets = [(row, col) for row in range(half_window_size) for col in range(half_window_size) if not row == half_window_size + 1 == col]
      for (row, col) in offsets:
          census = (census << 1) | (image[row:row + rows - half_window_size * 2, col:col + cols - half_window_size * 2] >= center_pixels)
      return census

  def column_cost(left_col, right_col):
      """
      Column-wise Hamming edit distance
      Also see https://www.youtube.com/watch?v=kxsvG4sSuvA&feature=youtu.be&t=1032
      :param left: numpy.ndarray(shape(Mx1), dtype=numpy.uint)
      :param right: numpy.ndarray(shape(Mx1), dtype=numpy.uint)
      :return: numpy.ndarray(shape(Mx1), dtype=numpy.uint)
      >>> image = np.array([ [50, 70, 80], [90, 100, 110], [60, 120, 150] ])
      >>> left = transform(image)
      >>> image = np.array([ [60, 75, 85], [115, 110, 105], [70, 130, 170] ])
      >>> right = transform(image)
      >>> column_cost(left, right)[0, 0]
      2
      """
      return np.sum(np.unpackbits(np.bitwise_xor(left_col, right_col), axis=1), axis=1).reshape(left_col.shape[0], left_col.shape[1])

  def cost(left, right, window_size=3, disparity=0):
      """
      Compute cost difference between left and right grayscale images. Disparity value can be used to assist with evaluating stereo
      correspondence.
      :param left: numpy.ndarray(shape=(MxN), dtype=numpy.uint8)
      :param right: numpy.ndarray(shape=(MxN), dtype=numpy.uint8)
      :param window_size: int odd-valued
      :param disparity: int
      :return:
      """
      ct_left = transform(left, window_size=window_size)
      ct_right = transform(right, window_size=window_size)
      rows, cols = ct_left.shape
      C = np.full(shape=(rows, cols), fill_value=0)
      for col in range(disparity, cols):
          C[:, col] = column_cost(
              ct_left[:, col:col + 1],
              ct_right[:, col - disparity:col - disparity + 1]
          ).reshape(ct_left.shape[0])
      return C

  def norm(image):
      return cv2.normalize(image, dst=None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)


  resize_pct = 0.5
  ndisp = 330
  ndisp *= resize_pct
  left = cv2.imread(str(path) +'/im0.png', 0)
  right = cv2.imread(str(path) +'/im0.png', 0)
  left = cv2.resize(left, dsize=(0,0), fx=resize_pct, fy=resize_pct)
  right = cv2.resize(right, dsize=(0, 0), fx=resize_pct, fy=resize_pct)

  window_size = size
  ct_left = norm(transform(left, window_size))
  ct_right = norm(transform(right, window_size))

  ct_costs = []
  for exponent in range(0, 6):
      import math
      disparity = int(ndisp / math.pow(2, exponent))
      print(math.pow(2, exponent), disparity)
      ct_costs.append(norm(cost(left, right, window_size, disparity)))

  cv2.imwrite('output_census.png', np.vstack(np.hstack([ct_left, ct_right])))
  return ct_left

def ret_path():
  norm_dirs = []
  dirs = list(os.listdir(os.getcwd()))

  for ele in dirs:
      if '.' in ele or 'test' in ele or 'results' in ele:
          continue
      norm_dirs.append(ele)

  return norm_dirs

if  __name__ == "__main__":

  print ('Start of execution...')
  dirs = ret_path()
  ground_l = plt.imread(str(path) + 'mask0nocc.png')
  ground_r = plt.imread(str(path) + 'mask1nocc.png')
                                             
  for element in dirs:
    abs_val = abs_blockmatching(element, 7)
    rms_val = rms_blockmatching(element, 7)
    abs_rms_val = abs_rms_blockmatching(element, 7)
    zncc_val = zncc_blockmatching(element, 7)
    census_val = census_blockmatching(element, 7)

    # For test images, remove this part

    rms_error_l_abs =  np.sqrt(np.mean(np.abs(abs_val[:,50:-50] - ground_l[:,50:-50])**2))
    rms_error_r_abs =  np.sqrt(np.mean(np.abs(abs_val[:,50:-50] - ground_r[:,50:-50])**2))
    rms_error_l_rms =  np.sqrt(np.mean(np.abs(rms_val[:,50:-50] - ground_l[:,50:-50])**2))
    rms_error_r_rms =  np.sqrt(np.mean(np.abs(rms_val[:,50:-50] - ground_r[:,50:-50])**2))
    rms_error_l_abs_rms =  np.sqrt(np.mean(np.abs(abs_rms_val[:,50:-50] - ground_l[:,50:-50])**2))
    rms_error_r_abs_rms =  np.sqrt(np.mean(np.abs(abs_rms_val[:,50:-50] - ground_r[:,50:-50])**2))
    rms_error_l_zncc =  np.sqrt(np.mean(np.abs(zncc_val[:,50:-50] - ground_l[:,50:-50])**2))
    rms_error_r_zncc =  np.sqrt(np.mean(np.abs(zncc_val[:,50:-50] - ground_r[:,50:-50])**2))
    rms_error_l_census =  np.sqrt(np.mean(np.abs(census_val[:,50:-50] - ground_l[:,50:-50])**2))
    rms_error_r_census =  np.sqrt(np.mean(np.abs(census_val[:,50:-50] - ground_r[:,50:-50])**2))

    #

  print ('...End of execution.')
