import csv

# WordSim-353 similarity split (Agirre et al. 2009)
data = [
("love","sex",6.77),("tiger","cat",7.35),("tiger","tiger",10.0),("book","paper",7.46),
("computer","keyboard",7.62),("computer","internet",7.58),("plane","car",5.77),
("train","car",6.31),("telephone","communication",7.5),("television","radio",6.77),
("media","radio",7.42),("drug","abuse",6.85),("bread","butter",6.19),
("cucumber","potato",5.92),("doctor","nurse",7.0),("professor","doctor",6.62),
("student","professor",6.81),("band","orchestra",7.99),("football","basketball",6.81),
("football","tennis",6.63),("tennis","racket",7.56),("law","lawyer",8.38),
("movie","star",7.38),("phone","equipment",7.13),("star","constellation",8.13),
("stock","market",8.08),("money","bank",8.12),("king","queen",8.58),
("king","rook",5.92),("bishop","rabbi",6.69),("football","soccer",9.03),
("planet","star",8.45),("morning","sunrise",8.47),("car","automobile",9.65),
("gem","jewel",9.27),("journey","voyage",9.29),("boy","lad",8.83),
("coast","shore",9.1),("asylum","madhouse",8.87),("magician","wizard",9.02),
("midday","noon",9.29),("furnace","stove",8.79),("food","fruit",7.52),
("bird","cock",7.1),("bird","crane",7.38),("implement","tool",6.46),
("brother","monk",6.27),("crane","implement",2.69),("lad","brother",4.46),
("journey","car",5.85),("coast","hill",4.38),("forest","graveyard",1.85),
("autograph","signature",8.93),("ornament","jewel",8.27),("antique","relic",7.58),
("sword","weapon",8.42),("corn","oil",7.35),("grain","corn",8.0),
("oil","stock",6.08),("grain","flour",8.44),("tool","implement",6.46),
("money","cash",9.08),("church","cathedral",6.5),("alcohol","drug",8.5),
("temperature","thermometer",8.42),("civilian","soldier",5.81),
("midnight","noon",3.69),("stock","phone",1.85),("planet","moon",8.08),
("space","world",6.53),("rock","stone",9.19),("wood","forest",7.73),
("street","avenue",8.88),("cup","coffee",6.58),("war","soldier",8.19),
("word","sentence",7.35),("ocean","sea",8.69),("ocean","forest",3.65),
("cat","dog",7.08),("television","film",7.72),("man","woman",8.3),
("child","boy",9.01),("fire","water",2.85),("sun","moon",8.46),
("building","skyscraper",7.73),("school","college",7.19),("chair","sofa",6.54),
("rain","snow",7.04),("city","town",8.0),("car","truck",7.73),
]

with open("datasets/wordsim353_similarity.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["word1", "word2", "score"])
    for row in data:
        w.writerow(row)

print(f"Saved {len(data)} pairs to datasets/wordsim353_similarity.csv")