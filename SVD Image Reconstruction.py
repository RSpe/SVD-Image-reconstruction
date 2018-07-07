from matplotlib import pyplot
import re
import numpy as np
import pandas as pd


def read_pgm(filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return np.frombuffer(buffer,
                         dtype='u1' if int(maxval) < 256 else byteorder + 'u2',
                         count=int(width) * int(height),
                         offset=len(header)
                         ).reshape((int(height), int(width)))


def display_pgm(array):  # Shows unaltered photo
    print("1.\n")
    pyplot.imshow(array, pyplot.cm.gray)
    pyplot.title("A")
    pyplot.axis('off')
    pyplot.show()


def svd(image):  # Performing SVD to find UDV^T
    u, s, vt = np.linalg.svd(image)
    return u, s, vt


def u(u_matrix):  # Scaled for better looking picture (U)
    u_max = np.amax(u_matrix)
    u_min = np.amin(u_matrix)
    u_scaled = 255 * ((u_matrix - u_min) / (u_max - u_min))
    pyplot.imshow(u_scaled, pyplot.cm.gray)
    pyplot.title("U")
    pyplot.axis('off')
    pyplot.show()
    return u_matrix


def s(s_matrix):  # Scaled for better looking picture (D)
    s_max = np.amax(s_matrix)
    s_min = np.amin(s_matrix)
    s_scaled = 255 * ((s_matrix - s_min) / (s_max - s_min))
    pyplot.imshow(np.diag(s_scaled[:1]), pyplot.cm.gray)
    pyplot.title("D")
    pyplot.axis('off')
    pyplot.show()
    return s_matrix


def vt(vt_matrix):  # Scaled for better looking picture (V)
    vt_max = np.amax(vt_matrix)
    vt_min = np.amin(vt_matrix)
    vt_scaled = 255 * ((vt_matrix - vt_min) / (vt_max - vt_min))
    pyplot.imshow(vt_scaled, pyplot.cm.gray)
    pyplot.title("V^T")
    pyplot.axis('off')
    pyplot.show()
    return vt_matrix


def recon(u_matrix, s_matrix, vt_matrix,
          img):  # Reconstructing the image in parts to show the effect of gradually less compression
    print("2.\n")
    for i in range(1, 6):
        reconstruct = np.matrix(u_matrix[:, :i]) * np.diag(s_matrix[:i]) * np.matrix((vt_matrix[:i, :]))
        pyplot.imshow(reconstruct, cmap='gray')
        title = str(i)
        pyplot.title("p = " + title)
        pyplot.axis('off')
        pyplot.show()
    for i in range(10, 101, 10):
        reconstruct = np.matrix(u_matrix[:, :i]) * np.diag(s_matrix[:i]) * np.matrix(vt_matrix[:i, :])
        pyplot.imshow(reconstruct, cmap='gray')
        title = str(i)
        pyplot.title("p = " + title)
        pyplot.axis('off')
        pyplot.show()


def small_comp(u_matrix, s_matrix, vt_matrix,
               img):  # Finding the smallest iterationg before compression becomes negative
    for i in range(0, 100):
        reconstruct = np.matrix(u_matrix[:, :i]) * np.diag(s_matrix[:i]) * np.matrix(vt_matrix[:i, :])
        compression = 1 - ((i * img.shape[0]) + i + (i * img.shape[1])) / (img.shape[0] * img.shape[1])
        title = str(i)
        if compression < 0:
            print(
                "\n4. Negative values for compression occur as the p value gets large as a result of more pixels are being used to create the output image as there are in the original picture, this is because more matrix values are in the reconstruct image as the orignal and these values are still plotted. \nAt p = " + title + " the p value is the smallest when compression becomes negative (" + (
                "{:.1%}".format(compression)) + ")")
            break


def threec(img, u_matrix, s_matrix, vt_matrix):  # p, max error, mean error, compression % for all compression levels
    p_list = []
    emax = []
    emean = []
    comp = []
    for i in range(1, 6):
        reconstruct = np.matrix(u_matrix[:, :i]) * np.diag(s_matrix[:i]) * np.matrix((vt_matrix[:i, :]))
        p_list.append(str(i))
        max_err = str(np.amax(abs(img - reconstruct)))
        emax.append(str("{:.3f}".format(float(max_err))))
        mean_err = str(np.mean(abs(img - reconstruct)))
        emean.append(str("{:.3f}".format(float(mean_err))))
        compression_percent = 1 - ((i * img.shape[0]) + i + (i * img.shape[1])) / (img.shape[0] * img.shape[1])
        if float(compression_percent) < 0:
            comp.append("N/A")
        else:
            comp.append(str("{:.1%}".format(float(compression_percent))))
    for i in range(10, 101, 10):
        reconstruct = np.matrix(u_matrix[:, :i]) * np.diag(s_matrix[:i]) * np.matrix((vt_matrix[:i, :]))
        p_list.append(str(i))
        max_err = str(np.amax(abs(img - reconstruct)))
        emax.append(str("{:.3f}".format(float(max_err))))
        mean_err = str(np.mean(abs(img - reconstruct)))
        emean.append(str("{:.3f}".format(float(mean_err))))
        compression_percent = 1 - ((i * img.shape[0]) + i + (i * img.shape[1])) / (img.shape[0] * img.shape[1])
        if float(compression_percent) < 0:
            comp.append("N/A")
        else:
            comp.append(str("{:.1%}".format(float(compression_percent))))
    d = {"p = ": p_list, "Max error": emax, "Mean error": emean, "Compression": comp}
    print("\n3. ")
    print(pd.DataFrame(data=d))


def threed(img, u_matrix, s_matrix, vt_matrix, p_value):  # What compression im happy with and why
    print("\n5. The most compressed approximation that is acceptable for me is when p = " + str(
        p_value) + " . I feel that at a normal viewing distance between the user and the computer the compression past this point shows nearly no signs of improvement. If I was to can the viewing distance to be between me and a TV then I assume you can take a very low p value (high compression) as the image is so small that differences in pixels would be nearly impossible to spot.")
    reconstruct = np.matrix(u_matrix[:, :p_value]) * np.diag(s_matrix[:p_value]) * np.matrix(vt_matrix[:p_value, :])
    compression_percent = "{:.1%}".format(
        1 - ((p_value * img.shape[0]) + p_value + (p_value * img.shape[1])) / (img.shape[0] * img.shape[1]))
    mean_err = str(np.mean(abs(img - reconstruct)))
    print("Compression = " + str(compression_percent) + ". Mean error = " + str("{:.3f}".format(float(mean_err))))


def main():
    image = read_pgm("me2.pgm")
    display_pgm(image)
    u_matrix, s_matrix, vt_matrix = svd(image)
    u_pic = u(u_matrix)
    s_pic = s(s_matrix)
    vt_pic = vt(vt_matrix)
    p_values = recon(u_pic, s_pic, vt_pic, image)
    c_answer = threec(image, u_pic, s_pic, vt_pic)
    small_p = small_comp(u_pic, s_pic, vt_pic, image)
    d_answer = threed(image, u_pic, s_pic, vt_pic, 40)


main()