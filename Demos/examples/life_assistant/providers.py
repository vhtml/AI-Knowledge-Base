"""外部数据源：天气（Open-Meteo 公开接口）与时区。

天气接口不需要 API Key：
- 地理编码 https://geocoding-api.open-meteo.com/v1/search?name=...
- 预报     https://api.open-meteo.com/v1/forecast?latitude=..&longitude=..

离线模式用 FakeWeatherProvider 造一份确定性假数据，保证示例不联网也能跑通。
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from . import cities
from .models import ToolError

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
REQUEST_TIMEOUT = 10.0

# WMO 天气代码 → 中文描述
WEATHER_CODES: dict[int, str] = {
    0: "晴",
    1: "大部晴朗",
    2: "局部多云",
    3: "阴",
    45: "有雾",
    48: "雾凇",
    51: "小毛毛雨",
    53: "毛毛雨",
    55: "大毛毛雨",
    56: "冻毛毛雨",
    57: "强冻毛毛雨",
    61: "小雨",
    63: "中雨",
    65: "大雨",
    66: "冻雨",
    67: "强冻雨",
    71: "小雪",
    73: "中雪",
    75: "大雪",
    77: "米雪",
    80: "阵雨",
    81: "强阵雨",
    82: "暴雨",
    85: "阵雪",
    86: "强阵雪",
    95: "雷阵雨",
    96: "雷阵雨伴冰雹",
    99: "强雷阵雨伴冰雹",
}

FAKE_CONDITIONS = ("晴", "多云", "阴", "小雨", "阵雨", "小雪", "大雾")

TIMEZONE_ALIASES = {
    "北京时间": "Asia/Shanghai",
    "北京时": "Asia/Shanghai",
    "东八区": "Asia/Shanghai",
    "协调世界时": "UTC",
    "utc": "UTC",
    "gmt": "UTC",
}

WEEKDAYS = ("周一", "周二", "周三", "周四", "周五", "周六", "周日")


@dataclass
class WeatherReport:
    """一次天气查询的结果。"""

    city: str
    day: str
    condition: str
    unit: str = "celsius"
    date: str = ""
    temperature: float | None = None
    feels_like: float | None = None
    humidity: int | None = None
    wind_kmh: float | None = None
    temp_min: float | None = None
    temp_max: float | None = None
    source: str = ""

    @property
    def symbol(self) -> str:
        """温度单位符号。"""

        return "℉" if self.unit == "fahrenheit" else "℃"

    def _convert(self, value: float | None) -> float | None:
        """把摄氏度换成华氏度。"""

        if value is None or self.unit != "fahrenheit":
            return value

        return value * 9 / 5 + 32

    def render(self) -> str:
        """渲染成一句可以直接说给用户听的话。"""

        scope = "今天" if self.day == "today" else "明天"
        parts = [f"{self.city}{scope}{self.condition}"]

        temperature = self._convert(self.temperature)

        if temperature is not None:
            parts.append(f"{temperature:.1f}{self.symbol}")

        feels_like = self._convert(self.feels_like)

        if feels_like is not None:
            parts.append(f"体感 {feels_like:.1f}{self.symbol}")

        temp_min = self._convert(self.temp_min)
        temp_max = self._convert(self.temp_max)

        if temp_min is not None and temp_max is not None:
            parts.append(f"{temp_min:.1f}{self.symbol}~{temp_max:.1f}{self.symbol}")

        if self.humidity is not None:
            parts.append(f"湿度 {int(self.humidity)}%")

        if self.wind_kmh is not None:
            parts.append(f"风速 {self.wind_kmh:.1f} km/h")

        if self.date:
            parts.append(self.date)

        return "，".join(parts) + "。"


class WeatherProvider(Protocol):
    """天气数据源的统一接口：在线用真的，离线用假的。"""

    name: str

    def get(self, city: str, day: str, unit: str) -> WeatherReport:
        """查询某个城市某一天的天气。"""


def _fetch_json(url: str) -> dict[str, Any]:
    """GET 一个 JSON 接口，把网络错误翻译成 ToolError。"""

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "life-assistant-demo/1.0"},
    )

    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
            body = response.read().decode("utf-8")
    except urllib.error.HTTPError as exc:
        raise ToolError(f"天气接口返回错误：HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise ToolError(f"联网获取天气失败：{exc.reason}") from None
    except TimeoutError:
        raise ToolError("联网获取天气超时") from None

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise ToolError("天气接口返回的不是合法 JSON") from None

    if not isinstance(payload, dict):
        raise ToolError("天气接口返回的数据格式异常")

    return payload


class OpenMeteoWeatherProvider:
    """真实数据源：Open-Meteo 公开接口（无需 API Key）。"""

    name = "Open-Meteo"

    def geocode(self, city: str) -> dict[str, Any]:
        """把城市名解析成经纬度。"""

        query = urllib.parse.urlencode(
            {"name": city, "count": 5, "language": "zh", "format": "json"}
        )
        payload = _fetch_json(f"{GEOCODE_URL}?{query}")
        results = payload.get("results") or []

        if not results:
            raise ToolError(f"没有找到城市「{city}」，换个常见城市名再试。")

        return results[0]

    def get(self, city: str, day: str = "today", unit: str = "celsius") -> WeatherReport:
        """查询天气。"""

        place = self.geocode(city)
        latitude = place.get("latitude")
        longitude = place.get("longitude")

        if latitude is None or longitude is None:
            raise ToolError(f"城市「{city}」没有查到经纬度")

        query = urllib.parse.urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,relative_humidity_2m,apparent_temperature,"
                    "weather_code,wind_speed_10m"
                ),
                "daily": "weather_code,temperature_2m_max,temperature_2m_min",
                "timezone": "auto",
                "forecast_days": 3,
            }
        )
        payload = _fetch_json(f"{FORECAST_URL}?{query}")

        daily = payload.get("daily") or {}
        current = payload.get("current") or {}
        dates = daily.get("time") or []
        index = 0 if day == "today" else 1

        if len(dates) <= index:
            raise ToolError("天气接口没有返回这几天的预报")

        report = WeatherReport(
            city=self._display_name(place, city),
            day=day,
            condition=self._condition(daily, index),
            unit=unit,
            date=str(dates[index]),
            source=self.name,
        )

        report.temp_min = self._daily_number(daily, "temperature_2m_min", index)
        report.temp_max = self._daily_number(daily, "temperature_2m_max", index)

        if index == 0:
            report.temperature = self._current_number(current, "temperature_2m")
            report.feels_like = self._current_number(current, "apparent_temperature")
            report.humidity = self._current_number(current, "relative_humidity_2m")
            report.wind_kmh = self._current_number(current, "wind_speed_10m")

        return report

    @staticmethod
    def _display_name(place: dict[str, Any], fallback: str) -> str:
        """挑一个中文城市名展示。"""

        name = str(place.get("name") or "")

        if name and any("一" <= char <= "鿿" for char in name):
            return name

        return fallback

    @staticmethod
    def _condition(daily: dict[str, Any], index: int) -> str:
        """把 WMO 代码翻译成中文。"""

        codes = daily.get("weather_code") or []

        if index >= len(codes):
            return "未知"

        return WEATHER_CODES.get(int(codes[index]), "未知")

    @staticmethod
    def _daily_number(daily: dict[str, Any], key: str, index: int) -> float | None:
        """从 daily 数组里取一个数。"""

        values = daily.get(key) or []

        if index >= len(values) or values[index] is None:
            return None

        return float(values[index])

    @staticmethod
    def _current_number(current: dict[str, Any], key: str) -> float | None:
        """从 current 里取一个数。"""

        value = current.get(key)

        if value is None:
            return None

        return float(value)


class FakeWeatherProvider:
    """离线假数据源：同一个城市同一天永远得到同样的结果。

    这样示例不联网也能跑通，而且自检可以断言具体数值。
    """

    name = "离线假数据"

    def get(self, city: str, day: str = "today", unit: str = "celsius") -> WeatherReport:
        """造一份确定性天气。"""

        digest = hashlib.sha256(f"{city}|{day}".encode("utf-8")).digest()
        temperature = float(8 + digest[0] % 22)
        condition = FAKE_CONDITIONS[digest[1] % len(FAKE_CONDITIONS)]
        humidity = int(40 + digest[2] % 50)
        wind = round(2 + (digest[3] % 120) / 10, 1)

        return WeatherReport(
            city=city,
            day=day,
            condition=condition,
            unit=unit,
            date="离线示例数据",
            temperature=temperature,
            feels_like=round(temperature - 1.5, 1),
            humidity=humidity,
            wind_kmh=wind,
            temp_min=round(temperature - 6, 1),
            temp_max=round(temperature + 3.5, 1),
            source=self.name,
        )


def resolve_timezone(timezone: str | None, city: str | None) -> str:
    """把时区参数或城市参数解析成一个 IANA 时区名。

    返回空字符串表示"用本机时区"。
    """

    if timezone:
        text = timezone.strip()

        if not text:
            return ""

        if text in TIMEZONE_ALIASES:
            return TIMEZONE_ALIASES[text]

        lowered = text.lower()

        if lowered in TIMEZONE_ALIASES:
            return TIMEZONE_ALIASES[lowered]

        return text

    if city:
        found = cities.resolve_city(city)

        if found is None:
            raise ToolError(
                f"不认识的城市「{city}」，查时间可以直接给时区，例如 Asia/Shanghai。"
            )

        return found.tz

    return ""


def current_moment(
    timezone: str | None,
    city: str | None,
    offset_days: int = 0,
) -> tuple[str, datetime]:
    """返回 (时区名, 目标时刻)。"""

    name = resolve_timezone(timezone, city)

    if name:
        try:
            zone = ZoneInfo(name)
        except (ZoneInfoNotFoundError, ValueError, KeyError):
            raise ToolError(
                f"无法识别的时区「{name}」，可以用「北京时间」「纽约」这类说法。"
            ) from None
    else:
        zone = datetime.now().astimezone().tzinfo
        name = local_timezone_name()

    moment = datetime.now(zone) + timedelta(days=offset_days)

    return name, moment


def local_timezone_name() -> str:
    """本机时区的 IANA 名。

    datetime.astimezone() 拿到的多半是 "CST" 这种缩写，
    读一下 /etc/localtime 软链接才能拿到 "Asia/Shanghai" 这种规范名。
    """

    try:
        link = os.path.realpath("/etc/localtime")
        marker = "/zoneinfo/"

        if marker in link:
            return link.split(marker, 1)[1]

    except OSError:
        pass

    return str(datetime.now().astimezone().tzinfo)


def format_moment(moment: datetime, output_format: str) -> str:
    """按指定的格式渲染时间。"""

    if output_format == "iso":
        return moment.isoformat()

    if output_format == "date":
        return moment.strftime("%Y-%m-%d")

    if output_format == "time":
        return moment.strftime("%H:%M")

    if output_format == "timestamp":
        return str(int(moment.timestamp()))

    weekday = WEEKDAYS[moment.weekday()]
    return moment.strftime("%Y-%m-%d %H:%M") + f"（{weekday}）"
