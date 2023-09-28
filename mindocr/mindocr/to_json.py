import json

data = {
    "apranax550.png": [
        {
            "transcription": "comprimbs",
            "points": [[52, 158], [98, 146], [100, 157], [55, 169]],
        },
        {
            "transcription": "polliculds",
            "points": [[98, 147], [139, 134], [142, 144], [101, 157]],
        },
        {
            "transcription": "secables",
            "points": [[139, 134], [179, 124], [182, 134], [141, 145]],
        },
        {
            "transcription": "sodique",
            "points": [[25, 120], [60, 109], [64, 121], [29, 133]],
        },
        {
            "transcription": "naproxene",
            "points": [[21, 107], [71, 94], [74, 108], [24, 120]],
        },
        {
            "transcription": "apranax",
            "points": [[17, 87], [86, 66], [90, 81], [21, 102]],
        },
        {"transcription": "550", "points": [[93, 64], [124, 55], [128, 70], [98, 79]]},
        {"transcription": "mg", "points": [[128, 54], [156, 50], [158, 66], [130, 70]]},
    ],
    "apyrosis.png": [
        {
            "transcription": "pyrosis",
            "points": [[274, 165], [326, 161], [343, 354], [291, 358]],
        },
        {
            "transcription": "suspens",
            "points": [[263, 168], [273, 168], [273, 209], [263, 209]],
        },
        {
            "transcription": "10m",
            "points": [[144, 150], [163, 149], [165, 202], [146, 203]],
        },
    ],
    "augmentin100.png": [
        {
            "transcription": "augmentin",
            "points": [[352, 381], [392, 405], [386, 416], [346, 391]],
        },
        {
            "transcription": "cnsion",
            "points": [[251, 341], [272, 354], [266, 363], [246, 350]],
        },
        {
            "transcription": "anthi",
            "points": [[217, 298], [272, 332], [262, 349], [207, 315]],
        },
        {
            "transcription": "buvable",
            "points": [[283, 294], [314, 313], [308, 324], [276, 305]],
        },
        {
            "transcription": "pour",
            "points": [[271, 275], [291, 286], [286, 297], [266, 286]],
        },
        {
            "transcription": "suspension",
            "points": [[240, 266], [281, 294], [274, 305], [233, 277]],
        },
        {
            "transcription": "poudre",
            "points": [[246, 254], [273, 275], [265, 285], [238, 265]],
        },
        {
            "transcription": "125ugparmi",
            "points": [[336, 194], [394, 232], [385, 246], [327, 208]],
        },
        {
            "transcription": "100mg",
            "points": [[301, 162], [342, 196], [327, 214], [286, 180]],
        },
        {
            "transcription": "augmentin",
            "points": [[325, 128], [503, 249], [482, 280], [304, 159]],
        },
    ],
    "augmentin1g.png": [
        {
            "transcription": "adulte",
            "points": [[366, 276], [379, 274], [386, 330], [373, 332]],
        },
        {
            "transcription": "mentin",
            "points": [[379, 267], [398, 265], [406, 330], [387, 332]],
        },
        {
            "transcription": "augm",
            "points": [[373, 207], [398, 207], [398, 277], [373, 277]],
        },
        {
            "transcription": "1q125n",
            "points": [[357, 211], [368, 210], [372, 250], [361, 251]],
        },
    ],
    "ciptadine.png": [
        {
            "transcription": "hikma",
            "points": [[248, 368], [321, 323], [334, 345], [262, 390]],
        },
        {
            "transcription": "4mg",
            "points": [[324, 237], [345, 230], [360, 277], [339, 284]],
        },
        {
            "transcription": "ciptadine",
            "points": [[389, 201], [410, 190], [452, 265], [431, 277]],
        },
    ],
    "clamoxyl1g.png": [
        {
            "transcription": "sibles",
            "points": [[341, 329], [377, 338], [374, 349], [338, 340]],
        },
        {
            "transcription": "disper",
            "points": [[307, 321], [343, 330], [340, 341], [304, 332]],
        },
        {
            "transcription": "comprimes",
            "points": [[242, 307], [305, 322], [302, 332], [239, 317]],
        },
        {
            "transcription": "disporad",
            "points": [[338, 256], [375, 266], [372, 276], [335, 266]],
        },
        {
            "transcription": "comprime",
            "points": [[345, 248], [382, 259], [380, 267], [343, 256]],
        },
        {
            "transcription": "clamoxyl",
            "points": [[255, 166], [407, 199], [399, 234], [248, 201]],
        },
    ],
    "clamoxyl500.png": [
        {
            "transcription": "getules",
            "points": [[263, 327], [298, 336], [295, 349], [260, 340]],
        },
        {
            "transcription": "500",
            "points": [[222, 264], [339, 296], [328, 335], [211, 303]],
        },
        {
            "transcription": "500ms",
            "points": [[316, 237], [368, 256], [361, 274], [309, 255]],
        },
        {
            "transcription": "clamoxyl",
            "points": [[245, 176], [369, 206], [361, 238], [237, 208]],
        },
    ],
    "clavor.png": [],
    "colchicine.png": [
        {
            "transcription": "audiupiod",
            "points": [[239, 357], [396, 344], [398, 369], [241, 382]],
        },
        {
            "transcription": "6w1",
            "points": [[241, 338], [270, 338], [270, 354], [241, 354]],
        },
        {
            "transcription": "wnd",
            "points": [[271, 337], [303, 334], [304, 345], [272, 348]],
        },
        {
            "transcription": "mexodo",
            "points": [[303, 331], [352, 328], [353, 343], [304, 346]],
        },
        {
            "transcription": "seconma",
            "points": [[278, 275], [352, 268], [353, 279], [279, 286]],
        },
        {
            "transcription": "ieumpslexer",
            "points": [[295, 266], [353, 262], [353, 272], [295, 276]],
        },
    ],
    "depakine.png": [
        {
            "transcription": "strop",
            "points": [[242, 385], [271, 380], [273, 392], [244, 397]],
        },
        {
            "transcription": "syrup",
            "points": [[274, 379], [308, 372], [310, 384], [276, 391]],
        },
        {
            "transcription": "sanofi",
            "points": [[154, 371], [169, 367], [184, 427], [169, 431]],
        },
        {
            "transcription": "diun",
            "points": [[373, 294], [384, 293], [386, 313], [375, 314]],
        },
        {
            "transcription": "depakine",
            "points": [[401, 220], [435, 212], [463, 327], [430, 336]],
        },
    ],
    "dulcamara.png": [
        {
            "transcription": "dulcamar",
            "points": [[342, 290], [389, 286], [390, 298], [343, 302]],
        }
    ],
    "dulcamarach.png": [
        {
            "transcription": "dulcamara",
            "points": [[294, 309], [350, 306], [351, 321], [295, 324]],
        }
    ],
    "fortalgic.png": [
        {
            "transcription": "comprimespenicutes",
            "points": [[434, 348], [513, 344], [513, 357], [434, 361]],
        },
        {
            "transcription": "paracdeomel",
            "points": [[324, 331], [369, 328], [370, 339], [325, 342]],
        },
        {
            "transcription": "coddine",
            "points": [[393, 326], [427, 324], [428, 335], [394, 337]],
        },
        {
            "transcription": "300mg",
            "points": [[329, 320], [366, 320], [366, 331], [329, 331]],
        },
        {
            "transcription": "25mg",
            "points": [[395, 316], [426, 316], [426, 327], [395, 327]],
        },
        {
            "transcription": "fortalaic",
            "points": [[298, 281], [483, 270], [485, 304], [300, 315]],
        },
    ],
    "gardénal100.png": [
        {
            "transcription": "sanofia",
            "points": [[360, 315], [414, 301], [417, 314], [363, 328]],
        },
        {
            "transcription": "mirocake",
            "points": [[187, 312], [221, 303], [224, 314], [190, 323]],
        },
        {
            "transcription": "forotox",
            "points": [[217, 303], [248, 296], [251, 307], [220, 314]],
        },
        {
            "transcription": "pherotarional",
            "points": [[178, 276], [221, 264], [224, 275], [181, 287]],
        },
        {
            "transcription": "phonoberbask",
            "points": [[221, 264], [266, 251], [269, 262], [224, 275]],
        },
        {
            "transcription": "gardenal",
            "points": [[171, 253], [253, 235], [257, 256], [175, 274]],
        },
        {
            "transcription": "100",
            "points": [[261, 233], [297, 225], [301, 246], [265, 253]],
        },
        {
            "transcription": "mg",
            "points": [[297, 224], [330, 220], [333, 241], [299, 245]],
        },
    ],
    "glucovance1000.png": [
        {
            "transcription": "merck",
            "points": [[181, 326], [196, 336], [159, 396], [144, 386]],
        },
        {
            "transcription": "0mghoigl",
            "points": [[336, 293], [358, 307], [299, 396], [278, 382]],
        },
        {
            "transcription": "gubenclamide",
            "points": [[280, 287], [291, 294], [260, 345], [249, 338]],
        },
        {
            "transcription": "glucovance",
            "points": [[384, 211], [402, 222], [356, 298], [338, 287]],
        },
    ],
    "glucovance500.png": [
        {
            "transcription": "30",
            "points": [[203, 372], [220, 369], [222, 383], [205, 385]],
        },
        {
            "transcription": "12x101",
            "points": [[220, 370], [245, 368], [246, 380], [221, 382]],
        },
        {
            "transcription": "ptableta",
            "points": [[242, 368], [273, 365], [274, 376], [243, 379]],
        },
        {
            "transcription": "ecomprimes",
            "points": [[269, 365], [313, 358], [315, 369], [271, 376]],
        },
        {
            "transcription": "merck",
            "points": [[396, 344], [459, 334], [461, 348], [398, 358]],
        },
        {
            "transcription": "oral",
            "points": [[197, 324], [214, 324], [214, 336], [197, 336]],
        },
        {
            "transcription": "uxhocosle",
            "points": [[211, 322], [264, 316], [266, 329], [213, 335]],
        },
        {
            "transcription": "nerhydrate",
            "points": [[203, 306], [244, 303], [245, 317], [204, 320]],
        },
        {
            "transcription": "de",
            "points": [[246, 305], [257, 305], [257, 314], [246, 314]],
        },
        {
            "transcription": "metformine",
            "points": [[256, 302], [304, 294], [306, 306], [258, 314]],
        },
        {
            "transcription": "metformia",
            "points": [[194, 297], [236, 292], [237, 303], [195, 308]],
        },
        {
            "transcription": "glibenclamide",
            "points": [[305, 294], [364, 284], [367, 298], [308, 308]],
        },
        {
            "transcription": "hydrochlorider",
            "points": [[235, 291], [297, 283], [299, 296], [237, 304]],
        },
        {
            "transcription": "aimecasted",
            "points": [[189, 275], [238, 267], [240, 280], [191, 288]],
        },
        {
            "transcription": "tablet",
            "points": [[233, 270], [261, 266], [262, 276], [235, 280]],
        },
        {
            "transcription": "ecomprene",
            "points": [[259, 267], [301, 260], [303, 271], [261, 278]],
        },
        {
            "transcription": "epellitole",
            "points": [[298, 260], [334, 254], [336, 266], [300, 272]],
        },
        {
            "transcription": "glucovance",
            "points": [[187, 253], [273, 240], [276, 257], [190, 270]],
        },
        {
            "transcription": "500",
            "points": [[282, 241], [309, 237], [312, 252], [284, 256]],
        },
        {
            "transcription": "mgl",
            "points": [[310, 238], [333, 238], [333, 253], [310, 253]],
        },
        {
            "transcription": "15",
            "points": [[330, 238], [341, 238], [341, 248], [330, 248]],
        },
        {
            "transcription": "mg",
            "points": [[344, 235], [363, 235], [363, 248], [344, 248]],
        },
    ],
    "ipproton.png": [
        {
            "transcription": "plozvadown",
            "points": [[245, 248], [259, 249], [255, 316], [241, 315]],
        },
        {
            "transcription": "no10dddi",
            "points": [[216, 240], [244, 241], [237, 400], [209, 399]],
        },
        {
            "transcription": "bwoz",
            "points": [[226, 163], [251, 166], [243, 228], [218, 225]],
        },
    ],
    "levocine.png": [
        {
            "transcription": "mg",
            "points": [[428, 330], [451, 333], [448, 351], [425, 348]],
        },
        {
            "transcription": "500",
            "points": [[384, 324], [429, 326], [428, 349], [383, 347]],
        },
        {
            "transcription": "consstute",
            "points": [[214, 327], [242, 327], [242, 337], [214, 337]],
        },
        {
            "transcription": "levofloxacine",
            "points": [[248, 309], [340, 309], [340, 319], [248, 319]],
        },
        {
            "transcription": "levocine",
            "points": [[242, 282], [349, 282], [349, 303], [242, 303]],
        },
    ],
    "levothyrox.png": [
        {
            "transcription": "microgrammes",
            "points": [[384, 283], [398, 279], [417, 339], [403, 343]],
        },
        {
            "transcription": "sccabl",
            "points": [[363, 256], [373, 252], [383, 280], [373, 284]],
        },
    ],
    "losar50.png": [
        {
            "transcription": "conreducated",
            "points": [[218, 319], [264, 317], [264, 328], [218, 330]],
        },
        {
            "transcription": "corpintspiuentally",
            "points": [[220, 314], [311, 308], [311, 318], [220, 324]],
        },
        {
            "transcription": "von",
            "points": [[169, 285], [185, 285], [185, 293], [169, 293]],
        },
        {
            "transcription": "losran",
            "points": [[290, 261], [318, 261], [318, 271], [290, 271]],
        },
        {
            "transcription": "potassique",
            "points": [[316, 259], [353, 257], [353, 267], [316, 269]],
        },
        {
            "transcription": "somg",
            "points": [[353, 256], [377, 256], [377, 267], [353, 267]],
        },
        {
            "transcription": "mg",
            "points": [[356, 235], [379, 238], [376, 256], [353, 253]],
        },
        {
            "transcription": "sar",
            "points": [[245, 229], [311, 225], [313, 256], [247, 260]],
        },
    ],
    "maxilase.png": [
        {
            "transcription": "sanofi",
            "points": [[310, 438], [355, 438], [355, 449], [310, 449]],
        },
        {
            "transcription": "125mi",
            "points": [[273, 416], [296, 416], [296, 427], [273, 427]],
        },
        {
            "transcription": "flecond",
            "points": [[268, 406], [291, 406], [291, 417], [268, 417]],
        },
        {
            "transcription": "entantsetadulties",
            "points": [[303, 389], [362, 389], [362, 400], [303, 400]],
        },
        {
            "transcription": "sirop",
            "points": [[313, 361], [341, 363], [340, 379], [312, 377]],
        },
        {
            "transcription": "dduceipimi",
            "points": [[305, 242], [366, 242], [366, 252], [305, 252]],
        },
        {
            "transcription": "alphasamylase",
            "points": [[305, 233], [376, 233], [376, 243], [305, 243]],
        },
        {
            "transcription": "de",
            "points": [[306, 219], [324, 219], [324, 230], [306, 230]],
        },
        {
            "transcription": "gorge",
            "points": [[326, 218], [372, 215], [373, 230], [327, 233]],
        },
        {
            "transcription": "maux",
            "points": [[306, 201], [346, 201], [346, 216], [306, 216]],
        },
        {
            "transcription": "aselxen",
            "points": [[252, 162], [294, 163], [291, 326], [249, 325]],
        },
    ],
    "panotile.png": [
        {
            "transcription": "tion",
            "points": [[265, 336], [276, 336], [276, 355], [265, 355]],
        },
        {
            "transcription": "zambon",
            "points": [[186, 314], [205, 312], [211, 371], [192, 373]],
        },
        {
            "transcription": "auriculaire",
            "points": [[242, 282], [255, 280], [264, 332], [251, 334]],
        },
        {
            "transcription": "panotile",
            "points": [[283, 276], [395, 257], [399, 278], [287, 297]],
        },
    ],
    "staphy.png": [
        {
            "transcription": "staphysagria",
            "points": [[287, 234], [347, 236], [347, 250], [287, 248]],
        }
    ],
    "sticta5chtg.png": [
        {
            "transcription": "mans",
            "points": [[378, 288], [405, 292], [403, 302], [376, 298]],
        },
        {
            "transcription": "fuelowind",
            "points": [[355, 282], [402, 279], [403, 290], [356, 293]],
        },
    ],
    "stilnox10.png": [
        {
            "transcription": "werhouters",
            "points": [[203, 338], [255, 329], [257, 341], [205, 350]],
        },
        {
            "transcription": "sanofls",
            "points": [[355, 328], [403, 316], [406, 326], [358, 338]],
        },
        {
            "transcription": "zolpidem",
            "points": [[201, 326], [254, 314], [257, 325], [204, 337]],
        },
        {
            "transcription": "10",
            "points": [[280, 308], [296, 303], [300, 315], [284, 320]],
        },
        {
            "transcription": "mg",
            "points": [[299, 303], [320, 301], [321, 314], [300, 316]],
        },
        {
            "transcription": "stilnox",
            "points": [[191, 290], [312, 267], [318, 299], [197, 322]],
        },
        {
            "transcription": "cid",
            "points": [[368, 251], [393, 245], [396, 256], [371, 262]],
        },
    ],
    "tranxene.png": [
        {
            "transcription": "sanofiy",
            "points": [[246, 358], [261, 359], [258, 413], [243, 412]],
        },
        {
            "transcription": "10mg",
            "points": [[367, 284], [397, 285], [395, 358], [365, 357]],
        },
        {
            "transcription": "assium",
            "points": [[354, 246], [361, 246], [361, 279], [354, 279]],
        },
        {
            "transcription": "tranxenee",
            "points": [[377, 161], [401, 161], [401, 277], [377, 277]],
        },
        {
            "transcription": "dorazi",
            "points": [[370, 163], [381, 163], [381, 194], [370, 194]],
        },
        {
            "transcription": "voleoralcloraluse",
            "points": [[324, 156], [337, 156], [334, 235], [321, 235]],
        },
    ],
    "unicorm3M.png": [
        {
            "transcription": "unicrom3m",
            "points": [[333, 290], [347, 283], [376, 340], [362, 347]],
        }
    ],
}

# Save the JSON data to a file
with open("data.json", "w") as file:
    json.dump(data, file, indent=4)
