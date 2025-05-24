key_maps = {
    # Minor keys
    "b" : {'i': 'b', 'ii-': 'c+', 'III': 'D', 'iv': 'e', 'V': 'F+', 'VI': 'G', 'VII': 'A',
            'I': 'B', '-II': 'C', 'ii': 'c+', 'IV': 'E', 'v': 'f+', 'vii': 'a'
    },
    "e" : {'i': 'e', 'ii-': 'f+', 'III': 'G', 'iv': 'a', 'V': 'B', 'VI': 'C', 'VII': 'D',
            'I': 'E', '-II': 'F', 'ii': 'f+', 'IV': 'A', 'v': 'b', 'vii': 'd'
    },
    "a" : {'i': 'a', 'ii-': 'b', 'III': 'C', 'iv': 'd', 'V': 'E', 'VI': 'F', 'VII': 'G',
            'I': 'A', '-II': 'B-', 'ii': 'b', 'IV': 'D', 'v': 'e', 'vii': 'g'
    },
    "d" : {'i': 'd', 'ii-': 'e', 'III': 'f', 'iv': 'g', 'V': 'A', 'VI': 'B-', 'VII': 'C',
            'I': 'D', '-II': 'E-', 'ii': 'e', 'IV': 'G', 'v': 'a', 'vii': 'c'
    },
    "g" : {'i': 'g', 'ii-': 'a', 'III': 'B-', 'iv': 'c', 'V': 'D', 'VI': 'E-', 'VII': 'F',
            'I': 'G', '-II': 'A-', 'ii': 'a', 'IV': 'C', 'v': 'd', 'vii': 'f'
    },
    "c" : {'i': 'c', 'ii-': 'd', 'III': 'E-', 'iv': 'f', 'V': 'G', 'VI': 'A-', 'VII': 'B-',
           'I': 'C', '-II': 'D-', 'ii': 'd', 'IV': 'F', 'v': 'g', 'vii': 'b-'
    },
    "f" : {'i': 'f', 'ii-': 'g', 'III': 'A-', 'iv': 'b-', 'V': 'C', 'VI': 'D-', 'VII': 'E-',
            'I': 'F', '-II': 'G-', 'ii': 'g', 'IV': 'B-', 'v': 'c', 'vii': 'e-'
    },
    "f+" : {'i': 'f+', 'ii-': 'g+', 'III': 'A', 'iv': 'b', 'V': 'C+', 'VI': 'D', 'VII': 'E', 
            'I': 'F+', '-II': 'G', 'ii': 'g+', 'IV': 'B', 'v': 'c+', 'vii': 'e'
    },   
     "c+" : {'i': 'c+', 'ii-': 'd+', 'III': 'E', 'iv': 'f+', 'V': 'G+', 'VI': 'A', 'VII': 'B', 
            'I': 'C+', '-II': 'D', 'ii': 'd+', 'IV': 'F+', 'v': 'g+', 'vii': 'b'
    },     
    "g+" : {'i': 'g+', 'ii-': 'a+', 'III': 'B', 'iv': 'c+', 'V': 'D+', 'VI': 'E', 'VII': 'F+', 
            'I': 'G+', '-II': 'A', 'ii': 'a+', 'IV': 'C+', 'v': 'd+', 'vii': 'f+'
    },       
    "d+" : {'i': 'd+', 'ii-': 'e+', 'III': 'F+', 'iv': 'g+', 'V': 'A+', 'VI': 'B', 'VII': 'C+', 
            'I': 'D+', '-II': 'E', 'ii': 'e+', 'IV': 'G+', 'v': 'a+', 'vii': 'c+'
    },  
    "a+" : {'i': 'a+', 'ii-': 'b+', 'III': 'C+', 'iv': 'd+', 'V': 'E+', 'VI': 'F+', 'VII': 'G+', 
            'I': 'A+', '-II': 'B', 'ii': 'b+', 'IV': 'D+', 'v': 'e+', 'vii': 'g+'
    },  
    # Major keys
    
    "B" : {'I': 'B', 'ii': 'c+', 'iii': 'd+', 'IV': 'E', 'V': 'F+', 'vi': 'g+', 'vii-': 'a+', 
           'i': 'b', 'iv' : 'e', 'v': 'f+', '-VII': 'A'
    },
    "E" : {'I': 'E', 'ii': 'f+', 'iii': 'g+', 'IV': 'A', 'V': 'B', 'vi': 'c+', 'vii-': 'd+', 
           'i': 'e', 'iv': 'a', 'v': 'b', '-VII': 'D'
    },
    "A" : {'I': 'A', 'ii': 'b', 'iii': 'c+', 'IV': 'D', 'V': 'E', 'vi': 'f+', 'vii-': 'g+', 
           'i': 'a', 'iv': 'd', 'v': 'e', '-VII': 'G'
    },
    "D" : {'I': 'D', 'ii': 'e', 'iii': 'F+', 'IV': 'G', 'V': 'A', 'vi': 'b', 'vii-': 'c+', 
           'i': 'd', 'iv': 'g', 'v': 'a', '-VII': 'C'
    },
    "G" : {'I': 'G', 'ii': 'a', 'iii': 'b', 'IV': 'C', 'V': 'D', 'vi': 'e', 'vii-': 'F+', 
           'i': 'g', 'iv': 'c', 'v':'d', '-VII': 'F'
    },
    "C" : {'I': 'C', 'ii': 'd', 'iii': 'e', 'IV': 'F', 'V': 'G', 'vi': 'a', 'vii-': 'b', 
           'i': 'c', 'iv': 'f', 'v': 'g', '-VII': 'B-'
    },
    "F" : {'I': 'F', 'ii': 'g', 'iii': 'a', 'IV': 'B-', 'V': 'C', 'vi': 'd', 'vii-': 'e', 
           'i': 'f', 'iv': 'b-', 'v': 'c', '-VII': 'E-'
    },
    "B-" : {'I': 'B-', 'ii': 'c', 'iii': 'd', 'IV': 'E-', 'V': 'F', 'vi': 'g', 'vii-': 'a', 
            'i': 'b-', 'iv': 'e-', 'v': 'f', '-VII': 'A-'
    },
    "E-" : {'I': 'E-', 'ii': 'f', 'iii': 'g', 'IV': 'A-', 'V': 'B-', 'vi': 'c', 'vii-': 'd', 
            'i': 'e-', 'iv': 'a-', 'v': 'b-', '-VII': 'D-'
    },
    "A-" : {'I': 'A-', 'ii': 'b-', 'iii': 'c', 'IV': 'D-', 'V': 'E-', 'vi': 'f', 'vii-': 'g', 
            'i': 'a-', 'iv': 'd-', 'v': 'e-', '-VII': 'G-'
    },
    "F+" : {'I': 'F+', 'ii': 'g+', 'iii': 'a+', 'IV': 'B', 'V': 'C+', 'vi': 'd+', 'vii-': 'e+', 
            'i': 'f+', 'iv': 'b', 'v': 'c+', '-VII': 'E'
    }
}

