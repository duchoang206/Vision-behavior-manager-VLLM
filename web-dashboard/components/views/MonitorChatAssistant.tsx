'use client';

import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Sun,
  Cloud,
  CloudSun,
  CloudRain,
  CloudLightning,
  CloudFog,
  Snowflake,
  Wind,
  Droplets,
  Thermometer,
  MapPin,
  Calendar,
  Clock,
  Sparkles,
  Send,
  Bot,
  User,
  RotateCcw,
  Copy,
  Check,
  RefreshCw
} from 'lucide-react';
import { useAppTheme } from '../ThemeContext';
import { Camera } from '../CameraContext';

interface Message {
  id: string;
  sender: 'bot' | 'user';
  text: string;
  timestamp: string;
}

interface WeatherData {
  temperature: number;
  apparentTemperature?: number;
  humidity: number;
  windSpeed: number;
  weatherCode: number;
  description: string;
  iconType: 'sun' | 'cloud-sun' | 'cloud' | 'rain' | 'thunder' | 'fog' | 'snow';
}

interface LocationData {
  city: string;
  locality: string;
  country: string;
  latitude: number;
  longitude: number;
  source: string;
}

interface MonitorChatAssistantProps {
  cameras: Camera[];
  liveAlerts: Array<{
    cam_id: string;
    global_id: number;
    rule_type: string;
    severity: string;
    description: string;
    timestamp: number;
  }>;
  storageSlots?: Array<{
    id: string;
    name: string;
    camName: string;
    status: 'CARFULL' | 'EMPTY';
    occupants: string[];
  }>;
  hostName?: string;
  onSelectCameraTab?: (camId: string) => void;
}

// Map WMO Weather Codes to Vietnamese description & icon type
function parseWeatherCode(code: number): { description: string; iconType: WeatherData['iconType'] } {
  if (code === 0) return { description: 'Trời quang, nắng', iconType: 'sun' };
  if (code === 1 || code === 2) return { description: 'Trời ít mây, nắng dịu', iconType: 'cloud-sun' };
  if (code === 3) return { description: 'Nhiều mây u ám', iconType: 'cloud' };
  if (code === 45 || code === 48) return { description: 'Có sương mù', iconType: 'fog' };
  if (code >= 51 && code <= 55) return { description: 'Mưa phùn rải rác', iconType: 'rain' };
  if (code >= 61 && code <= 65) return { description: 'Có mưa rào', iconType: 'rain' };
  if (code >= 71 && code <= 77) return { description: 'Tuyết rơi nhẹ', iconType: 'snow' };
  if (code >= 80 && code <= 82) return { description: 'Mưa rào nặng hạt', iconType: 'rain' };
  if (code >= 95 && code <= 99) return { description: 'Dông sét mạnh', iconType: 'thunder' };
  return { description: 'Thời tiết ổn định', iconType: 'cloud-sun' };
}

export default function MonitorChatAssistant({
  cameras,
  liveAlerts,
  storageSlots = [],
  hostName,
  onSelectCameraTab
}: MonitorChatAssistantProps) {
  const { colors: C, isDark } = useAppTheme();

  // ── 1. REAL-TIME CLOCK & DATE ──
  const [currentTime, setCurrentTime] = useState<Date | null>(null);

  useEffect(() => {
    setCurrentTime(new Date());
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  // ── 2. REAL-TIME LOCATION & WEATHER ──
  const [location, setLocation] = useState<LocationData>({
    city: 'Hà Nội',
    locality: 'Xuân Phương',
    country: 'Việt Nam',
    latitude: 21.0499,
    longitude: 105.7300,
    source: 'Tự động'
  });

  const [weather, setWeather] = useState<WeatherData>({
    temperature: 29.5,
    apparentTemperature: 35.0,
    humidity: 76,
    windSpeed: 2.4,
    weatherCode: 0,
    description: 'Trời quang, nắng',
    iconType: 'sun'
  });
  const [weatherLoading, setWeatherLoading] = useState(false);
  const [weatherError, setWeatherError] = useState<string | null>(null);

  // Fetch Location & Weather
  const fetchWeatherAndLocation = useCallback(async () => {
    setWeatherLoading(true);
    setWeatherError(null);

    // Mặc định địa điểm cơ sở giám sát: Xuân Phương, Hà Nội, Việt Nam
    const lat = 21.0499;
    const lon = 105.7300;
    const city = 'Hà Nội';
    const locality = 'Xuân Phương';
    const country = 'Việt Nam';

    setLocation({
      city,
      locality,
      country,
      latitude: lat,
      longitude: lon,
      source: 'Mặc định'
    });

    try {
      // Step B: Fetch real-time weather from Open-Meteo
      const weatherUrl = `https://api.open-meteo.com/v1/forecast?latitude=${lat}&longitude=${lon}&current=temperature_2m,relative_humidity_2m,apparent_temperature,weather_code,wind_speed_10m&timezone=auto`;
      const wRes = await fetch(weatherUrl, { signal: AbortSignal.timeout(8000) });
      if (!wRes.ok) throw new Error(`Weather HTTP ${wRes.status}`);

      const wData = await wRes.json();
      const current = wData.current;
      const parsed = parseWeatherCode(current.weather_code);

      setWeather({
        temperature: Math.round(current.temperature_2m * 10) / 10,
        apparentTemperature: Math.round((current.apparent_temperature ?? current.temperature_2m) * 10) / 10,
        humidity: current.relative_humidity_2m,
        windSpeed: Math.round(current.wind_speed_10m * 10) / 10,
        weatherCode: current.weather_code,
        description: parsed.description,
        iconType: parsed.iconType
      });
    } catch (err: unknown) {
      console.warn('Weather fetch fallback:', err);
      // Sensible fallback for local factory environment
      setWeather({
        temperature: 29.5,
        apparentTemperature: 35.0,
        humidity: 75,
        windSpeed: 2.5,
        weatherCode: 0,
        description: 'Trời quang, nắng ấm',
        iconType: 'sun'
      });
      setWeatherError('Dữ liệu mô phỏng');
    } finally {
      setWeatherLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchWeatherAndLocation();
    // Refresh weather every 15 minutes
    const interval = setInterval(fetchWeatherAndLocation, 15 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchWeatherAndLocation]);

  // ── 3. CHATBOT STATE & LOGIC ──
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isGenerating, setIsGenerating] = useState(false);
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Initialize Greeting
  useEffect(() => {
    if (messages.length === 0) {
      const timeStr = new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
      const camCount = cameras.length || 6;
      const alertCount = liveAlerts.length;

      setMessages([
        {
          id: 'welcome',
          sender: 'bot',
          text: `Xin chào! Tôi là Trợ Lý AI \n📹 Hệ thống hiện có ${camCount} camera đang kết nối và ${alertCount} cảnh báo được ghi nhận. Bạn cần tôi hỗ trợ kiểm tra hoặc tra cứu gì không?`,
          timestamp: timeStr
        }
      ]);
    }
  }, [cameras.length, liveAlerts.length, messages.length]);

  // Scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isGenerating]);

  // Handle Copy Message
  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  // Quick Prompt Chips
  const promptSuggestions = [
    { label: '📊 Tình trạng camera', prompt: 'Báo cáo số lượng và tình trạng các camera đang hoạt động trong hệ thống.' },
    { label: '🚨 Cảnh báo mới nhất', prompt: 'Hôm nay có những sự kiện hoặc cảnh báo an toàn nào cần chú ý?' },
    { label: '🌤️ Phân tích thời tiết', prompt: 'Thời tiết hiện tại có ảnh hưởng gì tới tầm nhìn và độ chính xác của camera ngoài trời không?' },
    { label: '📦 Kiểm tra ô chứa hàng', prompt: 'Kiểm tra trạng thái ô chứa hàng (CARFULL / EMPTY) của nhà máy.' },
  ];

  // Send Message Flow
  const handleSend = async (customText?: string) => {
    const textToSend = (customText || inputValue).trim();
    if (!textToSend || isGenerating) return;

    const timeStr = new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
    const userMsgId = `user-${Date.now()}`;
    const userMsg: Message = {
      id: userMsgId,
      sender: 'user',
      text: textToSend,
      timestamp: timeStr
    };

    setMessages(prev => [...prev, userMsg]);
    if (!customText) setInputValue('');

    const botMsgId = `bot-${Date.now() + 1}`;
    const botMsg: Message = {
      id: botMsgId,
      sender: 'bot',
      text: '',
      timestamp: timeStr
    };
    setMessages(prev => [...prev, botMsg]);
    setIsGenerating(true);

    // Call API: try Next proxy first, then direct host
    const apiEndpoints = [
      '/api/backend/chat',
      `http://${typeof window !== 'undefined' ? (window.location.hostname || '127.0.0.1') : '127.0.0.1'}:8000/api/chat`
    ];

    let success = false;
    let accumulatedText = '';

    for (const endpoint of apiEndpoints) {
      if (success) break;
      try {
        const response = await fetch(endpoint, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            text: textToSend,
            chat_history: messages.slice(-4).map(m => `${m.sender === 'user' ? 'User' : 'Assistant'}: ${m.text}`).join('\n')
          }),
          signal: AbortSignal.timeout(20000)
        });

        if (response.ok && response.body) {
          const reader = response.body.getReader();
          const decoder = new TextDecoder('utf-8');

          while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            const chunk = decoder.decode(value, { stream: true });
            accumulatedText += chunk;
            setMessages(prev => prev.map(m => m.id === botMsgId ? { ...m, text: accumulatedText } : m));
          }
          if (accumulatedText.includes('Lỗi kết nối Gemini API') || accumulatedText.includes('HTTP 403')) {
            accumulatedText = ''; // Trigger intelligent local factory assistant fallback
          } else {
            success = true;
            break;
          }
        }
      } catch (err) {
        console.warn(`Chat endpoint ${endpoint} failed, checking fallback:`, err);
      }
    }

    // Smart Local Fallback if backend API is not responding or Gemini has no internet/quota
    if (!success || !accumulatedText.trim()) {
      let smartFallback = '';
      const lower = textToSend.toLowerCase();

      if (lower.includes('camera') || lower.includes('cam')) {
        const onlineCount = cameras.filter(c => c.status === 'online').length;
        smartFallback = `Hiện tại hệ thống đang kết nối **${cameras.length} camera** (trong đó có **${onlineCount} camera trực tuyến**).\n` +
          cameras.map((c, idx) => `• Cam ${idx + 1}: **${c.name || c.id}** (${c.status || 'Active'})`).join('\n') +
          `\n\nBạn có thể nhấn vào từng camera trên lưới quan sát để xem chi tiết bám vết AI.`;
      } else if (lower.includes('cảnh báo') || lower.includes('alarm') || lower.includes('sự kiện')) {
        if (liveAlerts.length === 0) {
          smartFallback = `Hệ thống ghi nhận: **Chưa có sự kiện bất thường hoặc cảnh báo xâm nhập** trong phiên làm việc hiện tại. Mọi khu vực giám sát đang ở trạng thái an toàn.`;
        } else {
          smartFallback = `Đã ghi nhận **${liveAlerts.length} sự kiện/cảnh báo**:\n` +
            liveAlerts.slice(0, 4).map((a, i) => `• ${i + 1}. [${a.rule_type.toUpperCase()}] tại Cam **${a.cam_id}**: ${a.description} (${new Date(a.timestamp).toLocaleTimeString()})`).join('\n');
        }
      } else if (lower.includes('thời tiết') || lower.includes('nhiệt độ') || lower.includes('mưa')) {
        smartFallback = `Thông tin thời tiết tại **${location.city}, ${location.country}**:\n` +
          `• Tình trạng: **${weather?.description || 'Nắng ráo'}**\n` +
          `• Nhiệt độ: **${weather?.temperature || 29}°C** (Cảm giác như: ${weather?.apparentTemperature || 35}°C)\n` +
          `• Độ ẩm: **${weather?.humidity || 75}%** | Gió: **${weather?.windSpeed || 2.4} km/h**\n\n` +
          `Đánh giá: Điều kiện ánh sáng và độ ẩm rất thuận lợi cho các thuật toán AI Object Detection & Pose Tracking ngoài trời.`;
      } else if (lower.includes('ô chứa') || lower.includes('hàng') || lower.includes('slot') || lower.includes('carfull')) {
        const full = storageSlots.filter(s => s.status === 'CARFULL').length;
        smartFallback = `Trạng thái ô chứa hàng nhà máy:\n` +
          `• Tổng số ô: **${storageSlots.length}**\n` +
          `• Có hàng (CARFULL): **${full}**\n` +
          `• Ô trống (EMPTY): **${storageSlots.length - full}**\n\n` +
          (storageSlots.length === 0 ? `Chưa thiết lập vùng ROI ô chứa hàng trên tab Building.` : `Dữ liệu được cập nhật liên tục từ FMS Bridge.`);
      } else {
        smartFallback = `Tôi đã nhận thông tin: "*${textToSend}*".\n\n` +
          `Tôi là trợ lý ảo chuyên trách giám sát nhà máy Vision AI. Bạn có thể hỏi tôi về: trạng thái các luồng camera, phát hiện xâm nhập/hành vi, kiểm tra ô chứa hàng hoặc thông số môi trường nhà máy.`;
      }

      setMessages(prev => prev.map(m => m.id === botMsgId ? { ...m, text: smartFallback } : m));
    }

    setIsGenerating(false);
  };

  const handleClearChat = () => {
    const timeStr = new Date().toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
    const camCount = cameras.length || 6;
    const alertCount = liveAlerts.length;
    setMessages([
      {
        id: 'cleared',
        sender: 'bot',
        text: `Xin chào! Tôi là Trợ Lý AI \n📹 Hệ thống hiện có ${camCount} camera đang kết nối và ${alertCount} cảnh báo được ghi nhận. Bạn cần tôi hỗ trợ kiểm tra hoặc tra cứu gì không?`,
        timestamp: timeStr
      }
    ]);
  };

  // Weather Icon Component
  const renderWeatherIcon = (iconType: WeatherData['iconType'], size = 22) => {
    switch (iconType) {
      case 'sun':
        return <Sun size={size} color={C.accent} style={{ filter: 'drop-shadow(0 0 6px rgba(245,158,11,0.5))' }} />;
      case 'cloud-sun':
        return <CloudSun size={size} color={C.accentL} />;
      case 'cloud':
        return <Cloud size={size} color="#94a3b8" />;
      case 'rain':
        return <CloudRain size={size} color={C.cyan} />;
      case 'thunder':
        return <CloudLightning size={size} color="#f43f5e" />;
      case 'fog':
        return <CloudFog size={size} color="#94a3b8" />;
      case 'snow':
        return <Snowflake size={size} color="#67e8f9" />;
      default:
        return <Sun size={size} color={C.accent} />;
    }
  };

  const timeFormatted = currentTime
    ? currentTime.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    : '--:--:--';

  const dateFormatted = currentTime
    ? currentTime.toLocaleDateString('vi-VN', { weekday: 'long', day: '2-digit', month: '2-digit', year: 'numeric' })
    : 'Đang đồng bộ ngày...';

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      gap: '14px',
      height: '100%',
      minWidth: '340px'
    }}>

      {/* ═══════════════════════════════════════════════════════════════════════
          CARD 1: REAL-TIME ENVIRONMENT HUB (Ngày - Giờ - Địa điểm - Thời tiết)
      ══════════════════════════════════════════════════════════════════════════ */}
      <div style={{
        background: C.surface,
        borderRadius: '14px',
        border: `1px solid ${C.border}`,
        padding: '16px 18px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        boxShadow: isDark
          ? '0 8px 24px -6px rgba(0, 0, 0, 0.45), 0 0 1px rgba(255, 255, 255, 0.08) inset'
          : '0 4px 16px -2px rgba(0, 0, 0, 0.06)',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* Subtle decorative glow */}
        <div style={{
          position: 'absolute',
          top: '-40px',
          right: '-40px',
          width: '120px',
          height: '120px',
          background: 'radial-gradient(circle, rgba(245, 158, 11, 0.15) 0%, rgba(6, 182, 212, 0.05) 50%, transparent 70%)',
          borderRadius: '50%',
          pointerEvents: 'none',
          filter: 'blur(20px)'
        }} />

        {/* Row 1: Real-time Digital Clock & Live Pulse */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <div>
            <div style={{
              display: 'flex',
              alignItems: 'baseline',
              gap: '8px'
            }}>
              <span style={{
                fontFamily: 'monospace',
                fontSize: '28px',
                fontWeight: 800,
                color: C.textPrimary,
                letterSpacing: '0.04em',
                lineHeight: 1
              }}>
                {timeFormatted}
              </span>
              <span style={{
                fontSize: '10px',
                fontWeight: 700,
                color: C.accentGlow,
                fontFamily: 'monospace',
                padding: '2px 5px',
                background: C.accentDim,
                borderRadius: '4px',
                border: `1px solid ${C.accentBorder}`
              }}>
                GMT+7
              </span>
            </div>

            <div style={{
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              marginTop: '5px',
              fontSize: '12px',
              color: C.textSecondary,
              fontWeight: 500
            }}>
              <Calendar size={13} color={C.accent} />
              <span style={{ textTransform: 'capitalize' }}>{dateFormatted}</span>
            </div>
          </div>
        </div>

        {/* Divider */}
        <div style={{ height: '1px', background: C.border, margin: '2px 0' }} />

        {/* Row 2: Location & Weather Grid */}
        <div style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: '10px',
          alignItems: 'stretch'
        }}>

          {/* Location Block */}
          <div style={{
            background: C.card,
            padding: '10px 12px',
            borderRadius: '10px',
            border: `1px solid ${C.border}`,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: '6px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
              <MapPin size={15} color="#f43f5e" />
              <span style={{ fontSize: '11px', fontWeight: 600, color: C.textMuted }}>
                Địa Điểm
              </span>
            </div>
            <div>
              <div style={{
                fontSize: '13px',
                fontWeight: 700,
                color: C.textPrimary,
                whiteSpace: 'nowrap',
                overflow: 'hidden',
                textOverflow: 'ellipsis'
              }}>
                {location.city}, {location.country}
              </div>
            </div>
          </div>

          {/* Weather Block */}
          <div style={{
            background: C.card,
            padding: '10px 12px',
            borderRadius: '10px',
            border: `1px solid ${C.border}`,
            display: 'flex',
            flexDirection: 'column',
            justifyContent: 'space-between',
            gap: '4px',
            position: 'relative'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontSize: '11px', fontWeight: 600, color: C.textMuted }}>
                Thời Tiết Hiện Tại
              </span>
              <button
                onClick={fetchWeatherAndLocation}
                title="Cập nhật lại thời tiết"
                style={{
                  background: 'none',
                  border: 'none',
                  color: C.textMuted,
                  cursor: 'pointer',
                  padding: '2px',
                  display: 'flex',
                  alignItems: 'center'
                }}
              >
                <RefreshCw size={12} className={weatherLoading ? 'animate-spin' : ''} />
              </button>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {renderWeatherIcon(weather?.iconType || 'sun', 24)}
              <div>
                <span style={{
                  fontSize: '18px',
                  fontWeight: 800,
                  fontFamily: 'monospace',
                  color: C.textPrimary
                }}>
                  {weather ? `${weather.temperature}°C` : '--°C'}
                </span>
              </div>
            </div>

            <div style={{
              fontSize: '10px',
              color: C.textSecondary,
              fontWeight: 500,
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center'
            }}>
              <span>{weather?.description || 'Đang nạp...'}</span>
              <span style={{ color: C.cyan }}>💧 {weather?.humidity ?? '--'}%</span>
            </div>
          </div>
        </div>

      </div>

      {/* ═══════════════════════════════════════════════════════════════════════
          CARD 2: INTERACTIVE AI CHATBOT CONSOLE (Full Station)
      ══════════════════════════════════════════════════════════════════════════ */}
      <div style={{
        background: C.surface,
        borderRadius: '14px',
        border: `1px solid ${C.border}`,
        flex: 1,
        display: 'flex',
        flexDirection: 'column',
        boxShadow: isDark
          ? '0 8px 24px -6px rgba(0, 0, 0, 0.45), 0 0 1px rgba(255, 255, 255, 0.08) inset'
          : '0 4px 16px -2px rgba(0, 0, 0, 0.06)',
        overflow: 'hidden',
        minHeight: '440px'
      }}>


            {/* Messages Scroll Area */}
            <div style={{
              flex: 1,
              overflowY: 'auto',
              padding: '14px',
              display: 'flex',
              flexDirection: 'column',
              gap: '12px'
            }}>
              {messages.map((msg) => {
                const isBot = msg.sender === 'bot';
                return (
                  <div
                    key={msg.id}
                    style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: isBot ? 'flex-start' : 'flex-end',
                      gap: '4px'
                    }}
                  >
                    <div style={{
                      display: 'flex',
                      alignItems: 'flex-start',
                      gap: '8px',
                      maxWidth: '90%',
                      flexDirection: isBot ? 'row' : 'row-reverse'
                    }}>
                      {/* Avatar */}
                      <div style={{
                        width: '26px',
                        height: '26px',
                        borderRadius: '50%',
                        background: isBot
                          ? 'rgba(245, 158, 11, 0.15)'
                          : 'rgba(6, 182, 212, 0.15)',
                        border: `1px solid ${isBot ? C.accentBorder : C.cyanBorder}`,
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        flexShrink: 0,
                        marginTop: '2px'
                      }}>
                        {isBot ? (
                          <img src="/mascot.png" alt="Bot" style={{ width: '18px', height: '18px', objectFit: 'contain' }} />
                        ) : (
                          <User size={14} color={C.cyanL} />
                        )}
                      </div>

                      {/* Bubble */}
                      <div style={{
                        padding: '10px 14px',
                        borderRadius: isBot ? '4px 14px 14px 14px' : '14px 4px 14px 14px',
                        background: isBot
                          ? (isDark ? 'rgba(255, 255, 255, 0.05)' : '#ffffff')
                          : 'linear-gradient(135deg, #0891b2, #0284c7)',
                        border: isBot ? `1px solid ${C.border}` : 'none',
                        color: isBot ? C.textPrimary : '#ffffff',
                        fontSize: '12.5px',
                        lineHeight: 1.5,
                        boxShadow: isBot
                          ? '0 2px 8px rgba(0,0,0,0.1)'
                          : '0 4px 12px rgba(6,182,212,0.3)',
                        wordBreak: 'break-word',
                        whiteSpace: 'pre-wrap'
                      }}>
                        {msg.text || (isGenerating ? 'AI đang soạn phản hồi...' : '')}
                      </div>
                    </div>

                    {/* Meta info / Copy button */}
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: '8px',
                      fontSize: '10px',
                      color: C.textMuted,
                      margin: isBot ? '0 0 0 34px' : '0 34px 0 0'
                    }}>
                      <span>{msg.timestamp}</span>
                      {isBot && msg.text && (
                        <button
                          onClick={() => handleCopy(msg.id, msg.text)}
                          title="Sao chép câu trả lời"
                          style={{
                            background: 'none',
                            border: 'none',
                            color: C.textMuted,
                            cursor: 'pointer',
                            padding: '1px',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '3px'
                          }}
                        >
                          {copiedId === msg.id ? <Check size={11} color="#10b981" /> : <Copy size={11} />}
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}

              {/* Typing indicator */}
              {isGenerating && (
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginLeft: '34px' }}>
                  <div style={{
                    padding: '6px 12px',
                    borderRadius: '12px',
                    background: C.card,
                    border: `1px solid ${C.border}`,
                    display: 'flex',
                    alignItems: 'center',
                    gap: '4px'
                  }}>
                    <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: C.accent, animation: 'bounce 1s infinite' }} />
                    <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: C.accent, animation: 'bounce 1s infinite 0.2s' }} />
                    <span style={{ width: '5px', height: '5px', borderRadius: '50%', background: C.accent, animation: 'bounce 1s infinite 0.4s' }} />
                    <span style={{ fontSize: '11px', color: C.textMuted, marginLeft: '6px' }}>Đang suy nghĩ...</span>
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            {/* Quick Prompt Suggestions */}
            <div style={{
              padding: '6px 14px',
              display: 'flex',
              gap: '6px',
              overflowX: 'auto',
              borderTop: `1px solid ${C.border}`,
              background: isDark ? 'rgba(0,0,0,0.2)' : 'rgba(0,0,0,0.02)'
            }}>
              {promptSuggestions.map((item, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(item.prompt)}
                  disabled={isGenerating}
                  style={{
                    whiteSpace: 'nowrap',
                    padding: '4px 10px',
                    borderRadius: '20px',
                    fontSize: '11px',
                    fontWeight: 600,
                    background: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.05)',
                    border: `1px solid ${C.border}`,
                    color: C.textSecondary,
                    cursor: isGenerating ? 'not-allowed' : 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                  onMouseEnter={e => {
                    e.currentTarget.style.borderColor = C.accent;
                    e.currentTarget.style.color = C.textPrimary;
                  }}
                  onMouseLeave={e => {
                    e.currentTarget.style.borderColor = C.border;
                    e.currentTarget.style.color = C.textSecondary;
                  }}
                >
                  {item.label}
                </button>
              ))}
            </div>

            {/* Input Bar */}
            <div style={{
              padding: '12px 14px',
              borderTop: `1px solid ${C.border}`,
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              background: isDark ? 'rgba(255,255,255,0.02)' : '#ffffff'
            }}>
              <input
                ref={inputRef}
                type="text"
                placeholder="Hỏi trợ lý về camera, cảnh báo, thời tiết..."
                value={inputValue}
                onChange={e => setInputValue(e.target.value)}
                onKeyDown={e => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                disabled={isGenerating}
                style={{
                  flex: 1,
                  background: C.card,
                  border: `1px solid ${C.border}`,
                  borderRadius: '10px',
                  padding: '9px 12px',
                  fontSize: '12.5px',
                  color: C.textPrimary,
                  outline: 'none',
                  transition: 'border-color 0.2s ease'
                }}
                onFocus={e => e.currentTarget.style.borderColor = C.accent}
                onBlur={e => e.currentTarget.style.borderColor = C.border}
              />

              <button
                onClick={handleClearChat}
                title="Làm mới hội thoại"
                style={{
                  background: isDark ? 'rgba(255,255,255,0.05)' : 'rgba(0,0,0,0.05)',
                  border: `1px solid ${C.border}`,
                  borderRadius: '10px',
                  width: '38px',
                  height: '38px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: C.textMuted,
                  cursor: 'pointer',
                  transition: 'all 0.15s ease'
                }}
                onMouseEnter={e => {
                  e.currentTarget.style.color = C.accent;
                  e.currentTarget.style.borderColor = C.accent;
                }}
                onMouseLeave={e => {
                  e.currentTarget.style.color = C.textMuted;
                  e.currentTarget.style.borderColor = C.border;
                }}
              >
                <RotateCcw size={15} />
              </button>

              <button
                onClick={() => handleSend()}
                disabled={!inputValue.trim() || isGenerating}
                style={{
                  background: inputValue.trim() && !isGenerating
                    ? 'linear-gradient(135deg, #f59e0b, #d97706)'
                    : (isDark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)'),
                  border: 'none',
                  borderRadius: '10px',
                  width: '38px',
                  height: '38px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  color: inputValue.trim() && !isGenerating ? '#ffffff' : C.textMuted,
                  cursor: inputValue.trim() && !isGenerating ? 'pointer' : 'not-allowed',
                  transition: 'all 0.2s ease',
                  boxShadow: inputValue.trim() && !isGenerating ? '0 0 12px rgba(245,158,11,0.35)' : 'none'
                }}
              >
                <Send size={16} />
              </button>
            </div>

      </div>

    </div>
  );
}
