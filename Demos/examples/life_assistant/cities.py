"""内置城市表：中文城市名 → 英文检索名 + IANA 时区。

两处会用到它：
- 天气工具拿中文名去调 Open-Meteo 的地理编码接口；
- 时间工具把"北京时间""纽约时间"这类说法换算成时区。

表之外的城市并不是查不到——天气工具照样能通过地理编码接口找到，
只是拿不到这里的时区快捷方式。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class City:
    """一个内置城市。"""

    cn: str
    en: str
    tz: str


CITIES: tuple[City, ...] = (
    City("北京", "Beijing", "Asia/Shanghai"),
    City("上海", "Shanghai", "Asia/Shanghai"),
    City("广州", "Guangzhou", "Asia/Shanghai"),
    City("深圳", "Shenzhen", "Asia/Shanghai"),
    City("杭州", "Hangzhou", "Asia/Shanghai"),
    City("南京", "Nanjing", "Asia/Shanghai"),
    City("苏州", "Suzhou", "Asia/Shanghai"),
    City("成都", "Chengdu", "Asia/Shanghai"),
    City("重庆", "Chongqing", "Asia/Shanghai"),
    City("武汉", "Wuhan", "Asia/Shanghai"),
    City("西安", "Xian", "Asia/Shanghai"),
    City("天津", "Tianjin", "Asia/Shanghai"),
    City("厦门", "Xiamen", "Asia/Shanghai"),
    City("青岛", "Qingdao", "Asia/Shanghai"),
    City("长沙", "Changsha", "Asia/Shanghai"),
    City("郑州", "Zhengzhou", "Asia/Shanghai"),
    City("济南", "Jinan", "Asia/Shanghai"),
    City("沈阳", "Shenyang", "Asia/Shanghai"),
    City("大连", "Dalian", "Asia/Shanghai"),
    City("长春", "Changchun", "Asia/Shanghai"),
    City("哈尔滨", "Harbin", "Asia/Shanghai"),
    City("昆明", "Kunming", "Asia/Shanghai"),
    City("贵阳", "Guiyang", "Asia/Shanghai"),
    City("南宁", "Nanning", "Asia/Shanghai"),
    City("福州", "Fuzhou", "Asia/Shanghai"),
    City("合肥", "Hefei", "Asia/Shanghai"),
    City("南昌", "Nanchang", "Asia/Shanghai"),
    City("太原", "Taiyuan", "Asia/Shanghai"),
    City("石家庄", "Shijiazhuang", "Asia/Shanghai"),
    City("呼和浩特", "Hohhot", "Asia/Shanghai"),
    City("兰州", "Lanzhou", "Asia/Shanghai"),
    City("西宁", "Xining", "Asia/Shanghai"),
    City("银川", "Yinchuan", "Asia/Shanghai"),
    City("乌鲁木齐", "Urumqi", "Asia/Shanghai"),
    City("拉萨", "Lhasa", "Asia/Shanghai"),
    City("海口", "Haikou", "Asia/Shanghai"),
    City("三亚", "Sanya", "Asia/Shanghai"),
    City("香港", "Hong Kong", "Asia/Hong_Kong"),
    City("澳门", "Macao", "Asia/Macau"),
    City("台北", "Taipei", "Asia/Taipei"),
    City("东京", "Tokyo", "Asia/Tokyo"),
    City("首尔", "Seoul", "Asia/Seoul"),
    City("新加坡", "Singapore", "Asia/Singapore"),
    City("曼谷", "Bangkok", "Asia/Bangkok"),
    City("吉隆坡", "Kuala Lumpur", "Asia/Kuala_Lumpur"),
    City("雅加达", "Jakarta", "Asia/Jakarta"),
    City("马尼拉", "Manila", "Asia/Manila"),
    City("河内", "Hanoi", "Asia/Ho_Chi_Minh"),
    City("新德里", "New Delhi", "Asia/Kolkata"),
    City("孟买", "Mumbai", "Asia/Kolkata"),
    City("迪拜", "Dubai", "Asia/Dubai"),
    City("莫斯科", "Moscow", "Europe/Moscow"),
    City("伦敦", "London", "Europe/London"),
    City("巴黎", "Paris", "Europe/Paris"),
    City("柏林", "Berlin", "Europe/Berlin"),
    City("马德里", "Madrid", "Europe/Madrid"),
    City("罗马", "Rome", "Europe/Rome"),
    City("阿姆斯特丹", "Amsterdam", "Europe/Amsterdam"),
    City("苏黎世", "Zurich", "Europe/Zurich"),
    City("纽约", "New York", "America/New_York"),
    City("洛杉矶", "Los Angeles", "America/Los_Angeles"),
    City("旧金山", "San Francisco", "America/Los_Angeles"),
    City("西雅图", "Seattle", "America/Los_Angeles"),
    City("波士顿", "Boston", "America/New_York"),
    City("芝加哥", "Chicago", "America/Chicago"),
    City("多伦多", "Toronto", "America/Toronto"),
    City("温哥华", "Vancouver", "America/Vancouver"),
    City("悉尼", "Sydney", "Australia/Sydney"),
    City("墨尔本", "Melbourne", "Australia/Melbourne"),
    City("奥克兰", "Auckland", "Pacific/Auckland"),
    City("开普敦", "Cape Town", "Africa/Johannesburg"),
    City("开罗", "Cairo", "Africa/Cairo"),
)

# 别名 → 城市，常见口语说法
ALIASES: dict[str, str] = {
    "北京市": "北京",
    "上海市": "上海",
    "广州市": "广州",
    "深圳市": "深圳",
    "杭州市": "杭州",
    "成都市": "成都",
    "南京市": "南京",
    "武汉市": "武汉",
    "西安市": "西安",
    "天津市": "天津",
    "香港特别行政区": "香港",
    "纽约市": "纽约",
}

# 长名优先，避免"南京"被"南"之类的短片段截断
_BY_LENGTH = sorted(CITIES, key=lambda city: len(city.cn), reverse=True)


def resolve_city(name: str) -> City | None:
    """按中文名、别名或英文名精确查一个城市。"""

    text = name.strip()

    if not text:
        return None

    alias = ALIASES.get(text)

    if alias:
        text = alias

    for city in CITIES:
        if text == city.cn or text.lower() == city.en.lower():
            return city

    return None


def find_city(text: str) -> City | None:
    """在一段话里找出第一个出现的内置城市名。

    按城市名长度从长到短匹配，"哈尔滨"不会被"哈尔"这类前缀截断。
    """

    for city in _BY_LENGTH:
        if city.cn in text:
            return city

    for alias, target in ALIASES.items():
        if alias in text:
            found = resolve_city(target)

            if found is not None:
                return found

    return None
