"""The 24 UCF101 activities used by the project, with Vietnamese demo labels.

Chosen to look clearly different from each other and to cover sports, music,
group activities and everyday actions. Near-duplicates such as Basketball vs
BasketballDunk or FrontCrawl vs BreastStroke are deliberately left out.
"""

CLASSES = {
    "Archery": "Bắn cung",
    "BandMarching": "Diễu hành ban nhạc",
    "Basketball": "Bóng rổ",
    "Biking": "Đạp xe",
    "Billiards": "Bi-a",
    "Bowling": "Bowling",
    "Drumming": "Đánh trống",
    "Fencing": "Đấu kiếm",
    "GolfSwing": "Đánh golf",
    "HorseRiding": "Cưỡi ngựa",
    "IceDancing": "Khiêu vũ trên băng",
    "Kayaking": "Chèo thuyền kayak",
    "PizzaTossing": "Tung bánh pizza",
    "PlayingGuitar": "Chơi guitar",
    "PlayingPiano": "Chơi piano",
    "RockClimbingIndoor": "Leo núi trong nhà",
    "Skiing": "Trượt tuyết",
    "SkyDiving": "Nhảy dù",
    "SumoWrestling": "Đấu vật sumo",
    "Surfing": "Lướt sóng",
    "TableTennisShot": "Bóng bàn",
    "TaiChi": "Thái cực quyền",
    "Typing": "Gõ bàn phím",
    "WalkingWithDog": "Dắt chó đi dạo",
}

CLASS_NAMES = sorted(CLASSES)
CLASS_TO_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
