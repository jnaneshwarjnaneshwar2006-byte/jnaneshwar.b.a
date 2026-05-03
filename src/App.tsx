import { ImageSegmenter, FilesetResolver } from "@mediapipe/tasks-vision";
import React, { useState, useEffect, useRef, useCallback } from 'react';
import { 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer 
} from 'recharts';
import { 
  Camera, 
  Mic, 
  MicOff, 
  Video, 
  VideoOff, 
  Settings, 
  Activity, 
  MonitorUp,
  Wind,
  Smile, 
  Frown, 
  Meh, 
  Zap, 
  Info,
  Maximize2,
  Users,
  LayoutGrid,
  MoreVertical,
  PhoneOff,
  Check,
  CheckCheck,
  Plus,
  Search,
  Pin,
  PinOff,
  Play,
  Pause,
  RotateCcw,
  Brain
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { GoogleGenAI } from "@google/genai";
import ReactMarkdown from 'react-markdown';
import { cn } from './lib/utils';

// --- Constants & Types ---

const EMOJI_LIBRARY = {
  "Quick": ['👍', '👏', '🔥', '❤️', '😂', '😮'],
  "Expressions": ['😊', '😍', '🙄', '😱', '😴', '🥳', '🤔', '🤨', '🙄', '🤯'],
  "Status": ['💯', '✅', '❌', '⚠️', '🚀', '💎', '✨', '🎈', '🎉', '🔔'],
  "Objects": ['🍀', '🍎', '☕️', '🍕', '🍦', '🌈', '💻', '💡', '📚', '🎨']
};

const ANALYSIS_INTERVAL = 30000; // 30 seconds between AI checks to prevent rate limiting
const ANALYSIS_RETRY_DELAY = 60000; // 60 seconds wait after rate limit error
const GEMINI_MODEL = "gemini-3-flash-preview";

const getEmotionColor = (emotionStr: string) => {
  const e = emotionStr.toLowerCase();
  if (e.includes('joy') || e.includes('happ')) return 'text-green-400';
  if (e.includes('sad')) return 'text-red-400';
  if (e.includes('angry') || e.includes('frust')) return 'text-orange-400';
  if (e.includes('surprise')) return 'text-yellow-400';
  if (e.includes('neutral')) return 'text-cyan-400';
  if (e.includes('pensive')) return 'text-purple-400';
  if (e.includes('engaged')) return 'text-blue-400';
  return 'text-zinc-500';
};

const getEmotionRingColor = (emotionStr: string) => {
  const e = emotionStr.toLowerCase();
  if (e.includes('joy') || e.includes('happ')) return 'ring-green-500/40';
  if (e.includes('sad')) return 'ring-red-500/40';
  if (e.includes('angry') || e.includes('frust')) return 'ring-orange-500/40';
  if (e.includes('surprise')) return 'ring-yellow-500/40';
  if (e.includes('neutral')) return 'ring-cyan-500/30';
  if (e.includes('pensive')) return 'ring-purple-500/40';
  if (e.includes('engaged')) return 'ring-blue-500/40';
  return 'ring-white/5';
};

const AVATAR_OPTIONS = {
  top: ['ShortHairShortFlat', 'ShortHairShortRound', 'ShortHairTheCaesar', 'LongHairStraight', 'LongHairCurvy', 'LongHairBob', 'LongHairDreads', 'ShortHairSides', 'Hat', 'Turban'],
  accessories: ['Blank', 'Kurt', 'Prescription01', 'Prescription02', 'Round', 'Sunglasses', 'Wayfarers'],
  clothes: ['BlazerAndShirt', 'BlazerAndSweater', 'CollarAndSweater', 'GraphicShirt', 'Hoodie', 'Overall', 'ShirtCrewNeck', 'ShirtVNeck'],
  hairColor: ['Black', 'Auburn', 'Blonde', 'Brown', 'BrownDark', 'PastelPink', 'Platinum', 'Red', 'SilverGray'],
  clothingColor: ['Black', 'Blue01', 'Blue02', 'Blue03', 'Gray01', 'Gray02', 'Heather', 'PastelBlue', 'PastelGreen', 'Pink', 'Red', 'White'],
  eyes: ['Default', 'Close', 'Cry', 'Dizzy', 'EyeRoll', 'Happy', 'Hearts', 'Side', 'Squint', 'Surprised', 'Wink', 'WinkWacky'],
  mouth: ['Default', 'Concerned', 'Disbelief', 'Eating', 'Grimace', 'Sad', 'ScreamOpen', 'Serious', 'Smile', 'Tongue', 'Twinkle', 'Vomit']
};

const VIRTUAL_BACKGROUNDS = [
  { id: 'office', name: 'Tech Office', url: 'https://images.unsplash.com/photo-1497366216548-37526070297c?auto=format&fit=crop&q=80&w=1200' },
  { id: 'cyberpunk', name: 'Cyberpunk', url: 'https://images.unsplash.com/photo-1605810230434-7631ac76ec81?auto=format&fit=crop&q=80&w=1200' },
  { id: 'space', name: 'Deep Space', url: 'https://images.unsplash.com/photo-1451187580459-43490279c0fa?auto=format&fit=crop&q=80&w=1200' },
  { id: 'abstract', name: 'Neural Network', url: 'https://images.unsplash.com/photo-1620641788421-7a1c342ea42e?auto=format&fit=crop&q=80&w=1200' },
];

const getAvatarUrl = (seed: string, isLocal?: boolean, config?: any) => {
  if (isLocal && config) {
    const params = new URLSearchParams({
      top: config.top,
      accessories: config.accessories,
      hairColor: config.hairColor,
      clothes: config.clothes,
      clothingColor: config.clothingColor,
      eyes: config.eyes,
      mouth: config.mouth
    });
    return `https://api.dicebear.com/7.x/avataaars/svg?${params.toString()}`;
  }
  return `https://api.dicebear.com/7.x/avataaars/svg?seed=${seed}`;
};

type EmotionState = {
  primary: string;
  confidence: number;
  engagement: number; // 0-100
  notableTraits: string[];
  insight: string;
  timestamp?: string;
};

type ChatMessage = {
  id: string;
  senderId: string;
  text: string;
  timestamp: string;
  status?: 'sent' | 'delivered' | 'read';
};

type Reaction = {
  id: string;
  emoji: string;
  userId: string;
  timestamp: number;
};

type Participant = {
  id: string;
  name: string;
  status: string;
  avatarSeed: string;
  isLocal?: boolean;
};

// --- Mock Data ---

const MOCK_MESSAGES: ChatMessage[] = [
  { id: '1', senderId: '1', text: "The new engagement metrics look promising!", timestamp: "10:42 AM", status: 'read' },
  { id: '2', senderId: '2', text: "Agree, Sarah. Should we dive into the data?", timestamp: "10:43 AM", status: 'read' },
];

const INITIAL_EMOTION: EmotionState = {
  primary: "Neutral",
  confidence: 0,
  engagement: 0,
  notableTraits: [],
  insight: "Awaiting video feed analysis..."
};

const MOCK_PARTICIPANTS: Participant[] = [
  { id: 'local', name: 'You (Local Node)', status: 'Active', avatarSeed: 'local-user', isLocal: true },
  { id: '1', name: 'Sarah Jenkins', status: 'Joy (High Intensity)', avatarSeed: 'sarah' },
  { id: '2', name: 'Marcus King', status: 'Pensive / Neutral', avatarSeed: 'marcus' },
  { id: '3', name: 'Alex Rivera', status: 'Engaged', avatarSeed: 'alex' },
];

// --- Components ---

function FloatingEmojis({ userId, reactions }: { userId: string, reactions: Reaction[] }) {
  const userReactions = reactions.filter(r => r.userId === userId);
  
  return (
    <div className="absolute inset-0 pointer-events-none z-50 overflow-hidden">
      <AnimatePresence>
        {userReactions.map((r) => (
          <motion.div
            key={r.id}
            initial={{ y: "100%", x: "50%", opacity: 0, scale: 0.5 }}
            animate={{ 
              y: "-20%", 
              x: `${40 + Math.random() * 20}%`, 
              opacity: [0, 1, 1, 0, 0],
              scale: 1.5,
              rotate: [0, 10, -10, 0]
            }}
            exit={{ opacity: 0 }}
            transition={{ duration: 2.5, ease: "easeOut" }}
            className="absolute bottom-0 text-3xl filter drop-shadow-2xl"
          >
            {r.emoji}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

export default function App() {
  const [isCameraOn, setIsCameraOn] = useState(true);
  const [isMicOn, setIsMicOn] = useState(true);
  const [isNoiseCancellationActive, setIsNoiseCancellationActive] = useState(false);
  const [layout, setLayout] = useState<'mosaic' | 'stage' | 'cinema'>('stage');
  const [isFocusMode, setIsFocusMode] = useState(false); // Legacy toggle for sidebar
  const [isScreenSharing, setIsScreenSharing] = useState(false);
  const [pinnedParticipantId, setPinnedParticipantId] = useState<string | null>(null);
  const [sidebarTab, setSidebarTab] = useState<'insights' | 'chat' | 'profile'>('insights');
  const [avatarConfig, setAvatarConfig] = useState({
    top: 'ShortHairShortFlat',
    accessories: 'Blank',
    hairColor: 'Black',
    clothes: 'BlazerAndShirt',
    clothingColor: 'Blue03',
    eyes: 'Default',
    mouth: 'Default'
  });
  const [backgroundEffect, setBackgroundEffect] = useState<'none' | 'blur' | 'image'>('none');
  const [selectedBackground, setSelectedBackground] = useState(VIRTUAL_BACKGROUNDS[0].url);
  const [processedStream, setProcessedStream] = useState<MediaStream | null>(null);
  const [isRateLimited, setIsRateLimited] = useState(false);
  const [isAnalysisPaused, setIsAnalysisPaused] = useState(false);
  const [emotion, setEmotion] = useState<EmotionState>(INITIAL_EMOTION);
  const [history, setHistory] = useState<EmotionState[]>([]);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [participants] = useState<Participant[]>(MOCK_PARTICIPANTS);
  const [messages, setMessages] = useState<ChatMessage[]>(MOCK_MESSAGES);
  
  const mainParticipant = pinnedParticipantId 
    ? participants.find(p => p.id === pinnedParticipantId) 
    : (isCameraOn ? participants.find(p => p.isLocal) : participants.find(p => p.id === '1')) || participants[0];

  const [newMessage, setNewMessage] = useState("");
  const [typingParticipants, setTypingParticipants] = useState<string[]>([]);
  const [thinkingParticipants, setThinkingParticipants] = useState<string[]>([]);
  const [reactions, setReactions] = useState<Reaction[]>([]);
  const [isEmojiPickerOpen, setIsEmojiPickerOpen] = useState(false);
  const [customEmoji, setCustomEmoji] = useState("");
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const outputCanvasRef = useRef<HTMLCanvasElement>(null);
  const segmenterRef = useRef<ImageSegmenter | null>(null);
  const backgroundImgRef = useRef<HTMLImageElement | null>(null);
  const processingRef = useRef<number | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const screenStreamRef = useRef<MediaStream | null>(null);
  const screenVideoRef = useRef<HTMLVideoElement>(null);
  const audioCtxRef = useRef<AudioContext | null>(null);
  const noiseNodeRef = useRef<BiquadFilterNode | null>(null);

  const resetAnalysis = () => {
    setHistory([]);
    setEmotion(INITIAL_EMOTION);
    setIsAnalyzing(false);
    analysisLockRef.current = false;
  };

  // Load Background Image
  useEffect(() => {
    const img = new Image();
    img.src = selectedBackground;
    img.crossOrigin = "anonymous";
    img.onload = () => {
      backgroundImgRef.current = img;
    };
  }, [selectedBackground]);

  // Load Image Segmenter
  useEffect(() => {
    async function initSegmenter() {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.2/wasm"
        );
        const segmenter = await ImageSegmenter.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/image_segmenter/selfie_segmenter/float16/latest/selfie_segmenter.tflite",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          outputCategoryMask: true,
          outputConfidenceMasks: false
        });
        segmenterRef.current = segmenter;
        console.log("Segmenter loaded");
      } catch (err) {
        console.error("Failed to load segmenter:", err);
      }
    }
    initSegmenter();
  }, []);

  const processFrame = useCallback(() => {
    const video = videoRef.current;
    const outputCanvas = outputCanvasRef.current;
    const segmenter = segmenterRef.current;

    if (!video || !outputCanvas || !segmenter || !isCameraOn || backgroundEffect === 'none') {
      processingRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const ctx = outputCanvas.getContext('2d');
    if (!ctx) return;

    if (outputCanvas.width !== video.videoWidth) {
      outputCanvas.width = video.videoWidth;
      outputCanvas.height = video.videoHeight;
    }

    const startTimeMs = performance.now();
    
    segmenter.segmentForVideo(video, startTimeMs, (result) => {
      const mask = result.categoryMask?.getAsFloat32Array();
      if (!mask) return;

      const { width, height } = result.categoryMask!;
      
      // Draw background
      if (backgroundEffect === 'blur') {
        ctx.filter = 'blur(15px)';
        ctx.drawImage(video, 0, 0, width, height);
        ctx.filter = 'none';
      } else if (backgroundEffect === 'image' && backgroundImgRef.current) {
        ctx.drawImage(backgroundImgRef.current, 0, 0, width, height);
      }

      // Draw foreground
      const tempCanvas = document.createElement('canvas');
      tempCanvas.width = width;
      tempCanvas.height = height;
      const tempCtx = tempCanvas.getContext('2d')!;
      tempCtx.drawImage(video, 0, 0);

      const imageData = tempCtx.getImageData(0, 0, width, height);
      const data = imageData.data;

      for (let i = 0; i < mask.length; i++) {
        if (mask[i] < 0.1) {
          data[i * 4 + 3] = 0;
        }
      }
      tempCtx.putImageData(imageData, 0, 0);
      ctx.drawImage(tempCanvas, 0, 0);

      if (!processedStream && outputCanvas) {
        const stream = (outputCanvas as any).captureStream(30);
        setProcessedStream(stream);
      }
    });

    processingRef.current = requestAnimationFrame(processFrame);
  }, [isCameraOn, backgroundEffect, selectedBackground, processedStream]);

  useEffect(() => {
    if (isCameraOn && backgroundEffect !== 'none') {
      processingRef.current = requestAnimationFrame(processFrame);
    } else if (processingRef.current) {
      cancelAnimationFrame(processingRef.current);
    }
    return () => {
      if (processingRef.current) cancelAnimationFrame(processingRef.current);
    };
  }, [isCameraOn, backgroundEffect, processFrame]);

  // Initialize Camera
  useEffect(() => {
    if (isCameraOn) {
      navigator.mediaDevices.getUserMedia({ video: true, audio: true })
        .then(stream => {
          if (videoRef.current) {
            videoRef.current.srcObject = stream;
            streamRef.current = stream;
          }
        })
        .catch(err => {
          console.error("Camera access denied:", err);
          setIsCameraOn(false);
        });
    } else {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
    }
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
      }
      if (screenStreamRef.current) {
        screenStreamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, [isCameraOn]);

  // Audio Processing (Noise Cancellation)
  useEffect(() => {
    if (isMicOn && isNoiseCancellationActive && streamRef.current) {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      
      const ctx = audioCtxRef.current;
      const source = ctx.createMediaStreamSource(streamRef.current);
      
      // High-pass filter to remove low-frequency hum (simulating AI noise reduction)
      const filter = ctx.createBiquadFilter();
      filter.type = 'highpass';
      filter.frequency.value = 150; // Filter below 150Hz
      
      // Dynamic compression for voice clarity
      const compressor = ctx.createDynamicsCompressor();
      compressor.threshold.setValueAtTime(-24, ctx.currentTime);
      compressor.knee.setValueAtTime(40, ctx.currentTime);
      compressor.ratio.setValueAtTime(12, ctx.currentTime);
      compressor.attack.setValueAtTime(0, ctx.currentTime);
      compressor.release.setValueAtTime(0.25, ctx.currentTime);

      source.connect(filter);
      filter.connect(compressor);
      // In a real app, you'd route this to the WebRTC peer connection
      // For the demo, we just simulate the toggle state feedback
      
      noiseNodeRef.current = filter;
    } else if (audioCtxRef.current) {
      // Don't close the whole context to avoid pops, just disconnect
      noiseNodeRef.current?.disconnect();
    }
  }, [isNoiseCancellationActive, isMicOn]);

  // Screen Sharing Logic
  const toggleScreenShare = async () => {
    if (isScreenSharing) {
      if (screenStreamRef.current) {
        screenStreamRef.current.getTracks().forEach(track => track.stop());
      }
      screenStreamRef.current = null;
      setIsScreenSharing(false);
    } else {
      try {
        const stream = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: true
        });
        
        screenStreamRef.current = stream;
        if (screenVideoRef.current) {
          screenVideoRef.current.srcObject = stream;
        }

        stream.getVideoTracks()[0].onended = () => {
          setIsScreenSharing(false);
          screenStreamRef.current = null;
        };

        setIsScreenSharing(true);
      } catch (err) {
        console.error("Error sharing screen:", err);
      }
    }
  };

  // AI Analysis Loop
  const analysisLockRef = useRef(false);
  
  useEffect(() => {
    let intervalId: number;

    const analyzeFrame = async () => {
      // Check both state and ref lock
      if (!isCameraOn || !videoRef.current || !canvasRef.current || analysisLockRef.current || isRateLimited || isAnalysisPaused) return;

      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey) return;

      const ai = new GoogleGenAI({ apiKey });
      
      try {
        analysisLockRef.current = true;
        setIsAnalyzing(true);
        
        const canvas = canvasRef.current;
        const video = videoRef.current;
        const context = canvas.getContext('2d');
        if (!context) return;

        // Capture current frame
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const base64Image = canvas.toDataURL('image/jpeg', 0.5).split(',')[1];

        const prompt = `
          Analyze the facial expression and body count in this frame.
          Determine their primary emotion, a confidence score (0-1), an engagement level (0-100), 
          list 2-3 notable traits, and provide a short insight.
          
          Return valid JSON ONLY:
          {
            "primary": "Emotion",
            "confidence": 0.9,
            "engagement": 80,
            "notableTraits": ["trait1", "trait2"],
            "insight": "Human-readable insight"
          }
        `;

        const response = await ai.models.generateContent({
          model: GEMINI_MODEL,
          contents: [{ parts: [{ text: prompt }, { inlineData: { mimeType: "image/jpeg", data: base64Image } }] }],
          config: { responseMimeType: "application/json" }
        });

        const result = JSON.parse(response.text || "{}");
        
        setEmotion(result);
        setHistory(prev => [{ ...result, timestamp: new Date().toLocaleTimeString() }, ...prev].slice(0, 60));
      } catch (err: any) {
        const errMsg = err?.message || JSON.stringify(err);
        console.error("AI Analysis failed:", errMsg);
        
        if (errMsg.includes("429") || errMsg.includes("QUOTA_EXCEEDED") || errMsg.includes("RESOURCE_EXHAUSTED") || errMsg.includes("rate_limit")) {
          setIsRateLimited(true);
          setEmotion(prev => ({
            ...prev,
            insight: "System Capacity Reached. AI analysis on standby for 60s to sync with neural link..."
          }));
          
          setTimeout(() => {
            setIsRateLimited(false);
          }, ANALYSIS_RETRY_DELAY);
        }
      } finally {
        setIsAnalyzing(false);
        analysisLockRef.current = false;
      }
    };

    if (isCameraOn) {
      // Small jitter to prevent synchronized requests if multiple instances run
      const jitter = Math.random() * 2000;
      intervalId = window.setInterval(analyzeFrame, ANALYSIS_INTERVAL + jitter);
    }

    return () => clearInterval(intervalId);
  }, [isCameraOn, isRateLimited, isAnalysisPaused]); // Only depend on camera, rate-limit, and pause state

  const sendReaction = (emoji: string) => {
    const reaction: Reaction = {
      id: Date.now().toString() + Math.random(),
      emoji,
      userId: 'local',
      timestamp: Date.now()
    };
    setReactions(prev => [...prev, reaction]);
    
    // Auto-cleanup after 3 seconds
    setTimeout(() => {
      setReactions(prev => prev.filter(r => r.id !== reaction.id));
    }, 3000);
  };

  // Chat Simulation Logic
  useEffect(() => {
    if (sidebarTab === 'chat' && messages.length > 0) {
      const lastMsg = messages[messages.length - 1];
      if (lastMsg.senderId === 'local' && lastMsg.status === 'sent') {
        // Step 1: Delivered
        const dTimer = setTimeout(() => {
          setMessages(prev => prev.map(m => m.id === lastMsg.id ? { ...m, status: 'delivered' } : m));
        }, 800);

        // Step 2: Read
        const rTimer = setTimeout(() => {
          setMessages(prev => prev.map(m => m.id === lastMsg.id ? { ...m, status: 'read' } : m));
          
          // Phase 3: Reply Simulation
          const tTimer = setTimeout(() => {
            // Sarah starts thinking first
            setThinkingParticipants(['1']);
            
            const thinkingTimer = setTimeout(() => {
              setThinkingParticipants([]);
              setTypingParticipants(['1', '2']); // Sarah and Marcus typing together
              
              const replyTimer = setTimeout(() => {
                setTypingParticipants(prev => prev.filter(id => id !== '1'));
                const reply: ChatMessage = {
                  id: Date.now().toString(),
                  senderId: '1',
                  text: "I'm seeing a correlation between the engagement spikes and the current topic.",
                  timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                  status: 'read'
                };
                setMessages(prev => [...prev, reply]);
                
                // Keep Marcus typing for a bit more
                const marcusTimer = setTimeout(() => {
                  setTypingParticipants([]);
                  const marcusReply: ChatMessage = {
                    id: Date.now().toString() + 'm',
                    senderId: '2',
                    text: "Agreed. The neural feedback is quite telling here.",
                    timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                    status: 'read'
                  };
                  setMessages(prev => [...prev, marcusReply]);
                }, 2000);
                return () => clearTimeout(marcusTimer);
              }, 3500);
              return () => clearTimeout(replyTimer);
            }, 2500); // Thinking for 2.5s
            
            return () => clearTimeout(thinkingTimer);
          }, 1500);
          return () => clearTimeout(tTimer);
        }, 2200);

        return () => {
          clearTimeout(dTimer);
          clearTimeout(rTimer);
        };
      }
    }
  }, [messages, sidebarTab]);

  // Peer Reaction Simulation
  useEffect(() => {
    const timer = setInterval(() => {
      if (Math.random() > 0.7) {
        const p = participants[Math.floor(Math.random() * participants.length)];
        const emojis = ['👍', '👏', '🔥', '❤️', '😂', '😮'];
        const reaction: Reaction = {
          id: Date.now().toString() + Math.random(),
          emoji: emojis[Math.floor(Math.random() * emojis.length)],
          userId: p.id,
          timestamp: Date.now()
        };
        setReactions(prev => [...prev, reaction]);
        setTimeout(() => {
          setReactions(prev => prev.filter(r => r.id !== reaction.id));
        }, 3000);
      }
    }, 4500);
    return () => clearInterval(timer);
  }, [participants]);

  return (
    <div className="w-full h-screen bg-[#050506] flex p-4 gap-4 text-slate-200 overflow-hidden font-sans select-none">
      
      {/* MAIN VIDEO FEED AREA */}
      <motion.div 
        layout
        className="flex-1 flex flex-col gap-4 relative"
        transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
      >
        <div className={cn(
          "relative flex-1 rounded-3xl overflow-hidden transition-all duration-500",
          (layout === 'mosaic' && !isScreenSharing) ? "bg-transparent border-none" : "bg-[#121214] border border-white/5 shadow-2xl"
        )}>
          
          {/* GLOBAL OVERLAYS (Common to all layouts) */}
          <AnimatePresence>
            <motion.div 
              initial={{ y: -20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              className="absolute top-6 left-1/2 -translate-x-1/2 z-[40] flex items-center gap-3 bg-black/60 backdrop-blur-xl px-4 py-2 rounded-full border border-white/10 shadow-2xl pointer-events-none"
            >
              <div className={cn(
                "w-2 h-2 rounded-full animate-pulse shadow-[0_0_8px_rgba(34,211,238,0.8)]",
                isRateLimited ? "bg-orange-500" : isAnalyzing ? "bg-cyan-400" : "bg-green-500"
              )}></div>
              <span className="text-[10px] font-bold tracking-widest uppercase text-cyan-400 whitespace-nowrap">
                {isRateLimited ? "AI Cooldown" : isAnalyzing ? "Processing..." : "Real-time AI Active"}
              </span>
              <div className="h-3 w-[1px] bg-white/20 mx-1"></div>
              <span className="text-[10px] font-mono text-white/80 whitespace-nowrap uppercase">Conf: {Math.round(emotion.confidence * 100)}%</span>
            </motion.div>
          </AnimatePresence>

          {/* RENDER LOGIC BY LAYOUT */}
          
          {/* 1. SCREEN SHARING / PRESENTATION MODE */}
          {isScreenSharing ? (
            <div className="flex h-full gap-4 p-4">
              <div className="flex-[3] bg-black rounded-2xl overflow-hidden border border-white/10 relative group">
                <video 
                  ref={screenVideoRef} 
                  autoPlay 
                  playsInline 
                  className="w-full h-full object-contain"
                  onLoadedMetadata={() => {
                    if (screenVideoRef.current && screenStreamRef.current) {
                      screenVideoRef.current.srcObject = screenStreamRef.current;
                    }
                  }}
                />
                <div className="absolute top-4 left-4 px-3 py-1 bg-cyan-500 text-black text-[10px] font-bold uppercase rounded-full shadow-lg">Presenting Screen</div>
              </div>
              <div className="flex-1 flex flex-col gap-3 overflow-y-auto pr-1 scrollbar-hide">
                {participants.map(p => (
                  <div key={`side-${p.id}`} className="aspect-video bg-zinc-900 rounded-xl border border-white/5 overflow-hidden relative group">
                    {p.isLocal ? (
                      isCameraOn ? (
                        <video 
                          ref={videoRef} 
                          autoPlay 
                          playsInline 
                          muted 
                          className="w-full h-full object-cover scale-x-[-1]" 
                        />
                      ) : (
                        <div className="w-full h-full flex items-center justify-center">
                          <img src={getAvatarUrl(p.avatarSeed, p.isLocal, avatarConfig)} alt="" className="w-12 h-12 opacity-50" />
                        </div>
                      )
                    ) : (
                      <div className="w-full h-full bg-zinc-800 flex flex-col items-center justify-center p-4">
                        <img src={getAvatarUrl(p.avatarSeed, p.isLocal, avatarConfig)} alt="" className="w-12 h-12 mb-2" />
                        <span className="text-[8px] text-zinc-500 uppercase tracking-widest">{p.name}</span>
                      </div>
                    )}
                    <FloatingEmojis userId={p.id} reactions={reactions} />
                    <div className="absolute bottom-2 left-2 px-1.5 py-0.5 bg-black/60 rounded text-[8px] uppercase font-bold text-white transition-opacity opacity-0 group-hover:opacity-100">{p.name}</div>
                  </div>
                ))}
              </div>
            </div>
          ) : layout === 'mosaic' ? (
            /* 2. MOSAIC / GALLERY LAYOUT */
            <div className="grid grid-cols-2 gap-4 h-full">
              {participants.map((p) => (
                <div key={`mosaic-${p.id}`} className="bg-[#121214] rounded-2xl border border-white/5 overflow-hidden relative group shadow-lg">
                  {/* PIN BUTTON */}
                  <button 
                    onClick={() => setPinnedParticipantId(pinnedParticipantId === p.id ? null : p.id)}
                    className={cn(
                      "absolute top-4 right-4 z-20 p-2 rounded-full backdrop-blur-md transition-all",
                      pinnedParticipantId === p.id 
                        ? "bg-cyan-500 text-black shadow-lg shadow-cyan-900/40" 
                        : "bg-black/40 text-white/40 hover:text-white hover:bg-black/60 opacity-0 group-hover:opacity-100"
                    )}
                  >
                    {pinnedParticipantId === p.id ? <PinOff className="w-3.5 h-3.5" /> : <Pin className="w-3.5 h-3.5" />}
                  </button>

                  {p.isLocal ? (
                    isCameraOn ? (
                      <div className="w-full h-full relative">
                        <video 
                          ref={videoRef} 
                          autoPlay 
                          playsInline 
                          muted
                          className={cn("w-full h-full object-cover scale-x-[-1]", backgroundEffect !== 'none' && "hidden")} 
                        />
                        {backgroundEffect !== 'none' && (
                          <video 
                            ref={(el) => { if (el) el.srcObject = processedStream; }}
                            autoPlay 
                            playsInline 
                            muted
                            className="w-full h-full object-cover scale-x-[-1]" 
                          />
                        )}
                        <canvas ref={outputCanvasRef} className="hidden" />
                      </div>
                    ) : (
                      <div className="w-full h-full flex flex-col items-center justify-center bg-[#0a0a0b]">
                         <img src={getAvatarUrl(p.avatarSeed, p.isLocal, avatarConfig)} alt="" className="w-24 h-24 opacity-20 mb-4" />
                         <span className="text-[10px] text-zinc-600 uppercase font-mono tracking-widest">Feed Offline</span>
                      </div>
                    )
                  ) : (
                    <div className="w-full h-full bg-zinc-900/50 flex flex-col items-center justify-center relative overflow-hidden">
                       {/* Subtle Animated Glow for Activity */}
                       <div className="absolute inset-0 bg-gradient-to-tr from-cyan-500/5 to-transparent opacity-30"></div>
                       <img src={getAvatarUrl(p.avatarSeed, p.isLocal, avatarConfig)} alt="" className="w-24 h-24 mb-4 relative z-10" />
                       <div className="flex items-center gap-2 relative z-10">
                          <div className={cn("w-1.5 h-1.5 rounded-full", p.status.includes('High') ? "bg-green-500 animate-pulse" : "bg-zinc-600")}></div>
                          <span className="text-[10px] text-zinc-400 uppercase tracking-widest font-medium">{p.name}</span>
                       </div>
                    </div>
                  )}
                  <FloatingEmojis userId={p.id} reactions={reactions} />
                  <div className="absolute bottom-4 left-4 flex items-center gap-2">
                    <span className="px-2 py-1 bg-black/60 backdrop-blur-md rounded-lg text-[10px] font-bold uppercase tracking-tight text-white border border-white/5">
                      {p.name} {p.isLocal && "(You)"}
                    </span>
                    {(!p.isLocal && p.status.includes('High')) && (
                      <span className="px-2 py-1 bg-cyan-500 text-black rounded-lg text-[10px] font-bold uppercase flex items-center gap-1.5 shadow-lg shadow-cyan-900/20">
                         <Activity className="w-3 h-3" /> Peak Engagement
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          ) : layout === 'stage' ? (
            /* 3. STAGE LAYOUT (Speaker Focused) */
            <div className="h-full flex flex-col p-4 gap-4">
              <div className="flex-1 bg-black/40 rounded-2xl overflow-hidden border border-white/5 relative shadow-inner">
                {/* Active Speaker Area */}
                {mainParticipant?.isLocal ? (
                  isCameraOn ? (
                    <div className="w-full h-full relative">
                      <video 
                        ref={videoRef} 
                        autoPlay 
                        playsInline 
                        muted
                        className={cn("w-full h-full object-cover scale-x-[-1]", backgroundEffect !== 'none' && "hidden")} 
                      />
                      {backgroundEffect !== 'none' && (
                        <video 
                          ref={(el) => { if (el) el.srcObject = processedStream; }}
                          autoPlay 
                          playsInline 
                          muted
                          className="w-full h-full object-cover scale-x-[-1]" 
                        />
                      )}
                      <canvas ref={outputCanvasRef} className="hidden" />
                    </div>
                  ) : (
                    <div className="w-full h-full bg-zinc-900 flex flex-col items-center justify-center">
                      <img src={getAvatarUrl('local-user', true, avatarConfig)} alt="" className="w-48 h-48 mb-6" />
                      <span className="text-xl font-bold tracking-tight text-white">You (Feed Offline)</span>
                    </div>
                  )
                ) : (
                  <div className="w-full h-full bg-zinc-900 flex flex-col items-center justify-center">
                    <img src={getAvatarUrl(mainParticipant?.avatarSeed || 'default', false, avatarConfig)} alt="" className="w-48 h-48 mb-6" />
                    <span className="text-xl font-bold tracking-tight text-white">
                      {mainParticipant?.name} {pinnedParticipantId === mainParticipant?.id && "(Pinned)"}
                    </span>
                    <span className="text-sm text-cyan-400 uppercase font-mono mt-2 animate-pulse">
                      {pinnedParticipantId === mainParticipant?.id ? "FOCUSED FEED" : "Emotional Peak Detected"}
                    </span>
                  </div>
                )}
                <div className="absolute bottom-6 left-6 flex items-center gap-3">
                   <div className="w-10 h-10 rounded-full bg-cyan-500/20 border border-cyan-500/40 flex items-center justify-center">
                      <Activity className="w-5 h-5 text-cyan-400" />
                   </div>
                   <div>
                     <p className="text-xs font-bold text-white uppercase tracking-widest">
                       {pinnedParticipantId ? "Pinned Identity" : "Active Speaker Analysis"}
                     </p>
                     <p className="text-[10px] text-zinc-500 uppercase font-mono">{mainParticipant?.name}</p>
                   </div>
                </div>
                <FloatingEmojis userId={mainParticipant?.id || (isCameraOn ? 'local' : '1')} reactions={reactions} />
              </div>
              {/* Participant Strip */}
              <div className="h-24 flex items-center gap-3 overflow-x-auto scrollbar-hide pb-1">
                 {participants.map(p => (
                   <div 
                    key={`stage-strip-${p.id}`} 
                    className={cn(
                      "h-full aspect-video rounded-xl border flex-shrink-0 flex items-center justify-center relative group transition-all",
                      pinnedParticipantId === p.id ? "bg-cyan-500/20 border-cyan-500/50" : "bg-zinc-900 border-white/10"
                    )}
                   >
                      <img src={getAvatarUrl(p.avatarSeed, p.isLocal, avatarConfig)} alt="" className="w-10 h-10" />
                      
                      {/* PIN BUTTON IN STRIP */}
                      <button 
                        onClick={(e) => {
                          e.stopPropagation();
                          setPinnedParticipantId(pinnedParticipantId === p.id ? null : p.id);
                        }}
                        className={cn(
                          "absolute top-1 right-1 z-20 p-1.5 rounded-lg backdrop-blur-md transition-all",
                          pinnedParticipantId === p.id 
                            ? "bg-cyan-500 text-black" 
                            : "bg-black/60 text-white/40 hover:text-white opacity-0 group-hover:opacity-100"
                        )}
                      >
                        {pinnedParticipantId === p.id ? <PinOff className="w-3 h-3" /> : <Pin className="w-3 h-3" />}
                      </button>

                      <FloatingEmojis userId={p.id} reactions={reactions} />
                      <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-end p-2 rounded-xl">
                        <span className="text-[8px] font-bold text-white">{p.name}</span>
                      </div>
                   </div>
                 ))}
              </div>
            </div>
          ) : (
            /* 4. CINEMA LAYOUT (Classic Focus) */
            <div className="relative h-full overflow-hidden">
              <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-black/60 z-10 pointer-events-none"></div>
              
              {/* PIN TOGGLE HUD */}
              <button 
                onClick={() => setPinnedParticipantId(null)}
                className={cn(
                  "absolute top-6 right-6 z-30 px-3 py-1.5 rounded-full bg-black/60 backdrop-blur-xl border border-white/10 text-[10px] font-bold uppercase tracking-widest text-cyan-400 flex items-center gap-2 hover:bg-black/80 transition-all",
                  !pinnedParticipantId && "opacity-0 scale-90 pointer-events-none"
                )}
              >
                <PinOff className="w-3.5 h-3.5" /> Unpin Feed
              </button>

              {mainParticipant?.isLocal ? (
                isCameraOn ? (
                  <div className="w-full h-full relative">
                    <video 
                      ref={videoRef} 
                      autoPlay 
                      playsInline 
                      muted
                      className={cn("w-full h-full object-cover scale-x-[-1]", backgroundEffect !== 'none' && "hidden")} 
                    />
                    {backgroundEffect !== 'none' && (
                      <video 
                        ref={(el) => { if (el) el.srcObject = processedStream; }}
                        autoPlay 
                        playsInline 
                        muted
                        className="w-full h-full object-cover scale-x-[-1]" 
                      />
                    )}
                    <canvas ref={outputCanvasRef} className="hidden" />
                  </div>
                ) : (
                  <div className="absolute inset-0 flex flex-col items-center justify-center bg-[#0a0a0b]">
                    <div className="w-32 h-32 rounded-full border border-white/5 bg-zinc-900/50 flex items-center justify-center overflow-hidden mb-4 shadow-2xl ring-4 ring-cyan-500/10">
                      <img 
                        src={getAvatarUrl('local-user', true, avatarConfig)} 
                        alt="Local User" 
                        referrerPolicy="no-referrer"
                        className="w-full h-full object-cover opacity-50"
                      />
                    </div>
                    <VideoOff className="w-8 h-8 text-white/20" />
                    <span className="text-[10px] font-mono text-zinc-600 uppercase mt-4 tracking-widest leading-none">Stream Standby</span>
                  </div>
                )
              ) : (
                <div className="w-full h-full bg-zinc-900 flex flex-col items-center justify-center">
                  <img src={getAvatarUrl(mainParticipant?.avatarSeed || 'default', false, avatarConfig)} alt="" className="w-64 h-64 mb-8" />
                  <div className="flex flex-col items-center gap-2">
                    <span className="text-3xl font-black tracking-tighter uppercase text-white">{mainParticipant?.name}</span>
                    <span className="flex items-center gap-2 px-3 py-1 bg-cyan-500/10 border border-cyan-500/20 rounded-full text-[10px] font-mono text-cyan-400 uppercase tracking-widest">
                       Secure Neural Link Active
                    </span>
                  </div>
                </div>
              )}
              
              {/* CINEMA HUD OVERLAYS */}
              {mainParticipant?.isLocal && isCameraOn && (
                <motion.div 
                  animate={{ x: [0, 5, -5, 0], y: [0, -5, 5, 0] }}
                  transition={{ duration: 10, repeat: Infinity, ease: "easeInOut" }}
                  className="absolute top-1/4 left-1/4 z-20 p-4 border-2 border-cyan-500/40 rounded-xl bg-cyan-500/5 backdrop-blur-[2px]"
                >
                  <div className="absolute -top-1 -left-1 w-4 h-4 border-t-2 border-l-2 border-cyan-400"></div>
                  <div className="absolute -bottom-1 -right-1 w-4 h-4 border-b-2 border-r-2 border-cyan-400"></div>
                  <div className="flex flex-col gap-1">
                    <span className={cn("px-2 py-0.5 rounded text-[10px] font-bold uppercase", emotion.primary.toLowerCase().includes('hap') ? "bg-green-500/20 text-green-400" : "bg-cyan-500/20 text-cyan-400")}>{emotion.primary}</span>
                    <div className="flex flex-col gap-1.5 mt-2">
                      <div className="h-1.5 w-32 bg-white/10 rounded-full overflow-hidden"><motion.div key={emotion.engagement} initial={{ width: 0 }} animate={{ width: `${emotion.engagement}%` }} className="h-full bg-cyan-400 shadow-[0_0_8px_rgba(34,211,238,0.6)]" /></div>
                    </div>
                  </div>
                </motion.div>
              )}
              <FloatingEmojis userId={mainParticipant?.id || (isCameraOn ? 'local' : '1')} reactions={reactions} />
            </div>
          )}

          {/* SELF VIEW (MINI) */}
          <motion.div 
            animate={{ 
              scale: isFocusMode ? 0.8 : 1,
              opacity: isFocusMode ? 0.6 : 1
            }}
            className="absolute bottom-6 right-6 w-56 aspect-video bg-[#1e1e21] rounded-2xl border border-white/10 shadow-xl overflow-hidden ring-4 ring-black/40 z-20"
          >
            <div className="absolute inset-0 flex items-center justify-center opacity-20">
               <img 
                 src={getAvatarUrl('local-user', true, avatarConfig)} 
                 alt="Local User" 
                 referrerPolicy="no-referrer"
                 className="w-20 h-20"
               />
            </div>
            <div className="absolute inset-0 flex items-center justify-center">
               {isCameraOn ? (
                 <div className="flex flex-col items-center gap-1">
                   <div className="w-1 h-1 rounded-full bg-cyan-400 animate-ping"></div>
                   <span className="text-[10px] text-cyan-400 font-bold tracking-tighter uppercase">Capturing</span>
                 </div>
               ) : (
                 <VideoOff className="w-6 h-6 text-white/10" />
               )}
            </div>
            <div className="absolute bottom-2 left-2 px-2 py-1 bg-black/60 rounded text-[10px] font-bold uppercase tracking-tighter">Node_Local</div>
            <FloatingEmojis userId="local" reactions={reactions} />
          </motion.div>

          {/* CALL CONTROLS */}
          <div className="absolute bottom-6 left-1/2 -translate-x-1/2 flex items-center gap-4 z-30">
            <button 
              onClick={() => setIsMicOn(!isMicOn)}
              className={cn(
                "w-12 h-12 rounded-full border flex items-center justify-center transition-all",
                isMicOn ? "bg-white/5 border-white/10 hover:bg-white/10" : "bg-red-500/20 border-red-500/50 text-red-500"
              )}
            >
              {isMicOn ? <Mic className="w-5 h-5" /> : <MicOff className="w-5 h-5" />}
            </button>

            <button 
              onClick={() => setIsNoiseCancellationActive(!isNoiseCancellationActive)}
              className={cn(
                "w-12 h-12 rounded-full border flex items-center justify-center transition-all relative group",
                isNoiseCancellationActive ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-400 shadow-[0_0_15px_rgba(34,211,238,0.2)]" : "bg-white/5 border-white/10 hover:bg-white/10"
              )}
              title="Neural Noise Suppression"
            >
              <Wind className={cn("w-5 h-5", isNoiseCancellationActive && "animate-pulse")} />
              {isNoiseCancellationActive && (
                <div className="absolute -top-1 -right-1 w-3 h-3 bg-cyan-500 rounded-full border-2 border-[#050506]" />
              )}
            </button>
            <button className="w-16 h-16 rounded-full bg-red-600 border border-red-500 shadow-lg shadow-red-900/40 text-white flex items-center justify-center hover:scale-105 transition-all">
              <PhoneOff className="w-8 h-8" />
            </button>
            <button 
              onClick={() => setIsCameraOn(!isCameraOn)}
              className={cn(
                "w-12 h-12 rounded-full border flex items-center justify-center transition-all",
                isCameraOn ? "bg-white/5 border-white/10 hover:bg-white/10" : "bg-red-500/20 border-red-500/50 text-red-500"
              )}
            >
              {isCameraOn ? <Video className="w-5 h-5" /> : <VideoOff className="w-5 h-5" />}
            </button>

            <button 
              onClick={toggleScreenShare}
              className={cn(
                "w-12 h-12 rounded-full border flex items-center justify-center transition-all",
                isScreenSharing ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-400" : "bg-white/5 border-white/10 hover:bg-white/10"
              )}
            >
              <MonitorUp className="w-5 h-5" />
            </button>

            {/* LAYOUT CONTROLS */}
            <div className="w-[1px] h-8 bg-white/10 mx-2" />
            <div className="flex bg-black/40 backdrop-blur-md p-1 rounded-full border border-white/5">
              <button 
                onClick={() => setLayout('mosaic')}
                className={cn(
                  "p-2 rounded-full transition-all",
                  layout === 'mosaic' ? "bg-cyan-500/20 text-cyan-400" : "text-zinc-500 hover:text-zinc-300"
                )}
              >
                <LayoutGrid className="w-5 h-5" />
              </button>
              <button 
                onClick={() => setLayout('stage')}
                className={cn(
                  "p-2 rounded-full transition-all",
                  layout === 'stage' ? "bg-cyan-500/20 text-cyan-400" : "text-zinc-500 hover:text-zinc-300"
                )}
              >
                <Users className="w-5 h-5" />
              </button>
              <button 
                onClick={() => setLayout('cinema')}
                className={cn(
                  "p-2 rounded-full transition-all",
                  layout === 'cinema' ? "bg-cyan-500/20 text-cyan-400" : "text-zinc-500 hover:text-zinc-300"
                )}
              >
                <Maximize2 className="w-5 h-5" />
              </button>
            </div>

            {/* FOCUS MODE TOGGLE (Sidebar control) */}
            <button 
              onClick={() => setIsFocusMode(!isFocusMode)}
              className={cn(
                "w-12 h-12 rounded-full border flex items-center justify-center transition-all",
                isFocusMode ? "bg-cyan-500/20 border-cyan-500/50 text-cyan-400" : "bg-white/5 border-white/10 hover:bg-white/10"
              )}
            >
              <Activity className="w-5 h-5" />
            </button>

            {/* REACTION BAR */}
            <div className="flex items-center bg-black/40 backdrop-blur-xl p-1.5 rounded-full border border-white/5 shadow-2xl ml-2 relative">
              {EMOJI_LIBRARY.Quick.map(emoji => (
                <button 
                  key={emoji}
                  onClick={() => sendReaction(emoji)}
                  className="w-10 h-10 flex items-center justify-center text-xl hover:bg-white/5 rounded-full transition-all hover:scale-125"
                >
                  {emoji}
                </button>
              ))}
              
              <div className="w-[1px] h-6 bg-white/10 mx-1" />
              
              <button 
                onClick={() => setIsEmojiPickerOpen(!isEmojiPickerOpen)}
                className={cn(
                  "w-10 h-10 flex items-center justify-center text-zinc-400 hover:text-white hover:bg-white/5 rounded-full transition-all",
                  isEmojiPickerOpen && "bg-cyan-500/20 text-cyan-400"
                )}
              >
                <Plus className={cn("w-5 h-5 transition-transform", isEmojiPickerOpen && "rotate-45")} />
              </button>

              {/* EXPANDED EMOJI PICKER */}
              <AnimatePresence>
                {isEmojiPickerOpen && (
                  <motion.div 
                    initial={{ opacity: 0, y: 20, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 20, scale: 0.95 }}
                    className="absolute bottom-full mb-4 right-0 w-72 bg-[#121214]/95 backdrop-blur-2xl border border-white/10 rounded-3xl shadow-2xl p-4 z-50 overflow-hidden"
                  >
                    <div className="flex items-center justify-between mb-4">
                      <span className="text-[10px] font-bold uppercase tracking-widest text-zinc-500">Emoji Library</span>
                      <div className="flex items-center gap-2 bg-white/5 border border-white/10 rounded-lg px-2 py-1">
                        <input 
                          type="text" 
                          placeholder="Code..." 
                          value={customEmoji}
                          onChange={(e) => setCustomEmoji(e.target.value)}
                          className="w-12 bg-transparent text-[10px] focus:outline-none font-mono"
                          onKeyDown={(e) => {
                            if (e.key === 'Enter' && customEmoji.trim()) {
                              sendReaction(customEmoji.trim());
                              setCustomEmoji("");
                              setIsEmojiPickerOpen(false);
                            }
                          }}
                        />
                        <button 
                          onClick={() => {
                            if (customEmoji.trim()) {
                              sendReaction(customEmoji.trim());
                              setCustomEmoji("");
                              setIsEmojiPickerOpen(false);
                            }
                          }}
                          className="text-cyan-400 hover:text-cyan-300"
                        >
                          <Zap className="w-3 h-3" />
                        </button>
                      </div>
                    </div>

                    <div className="space-y-4 max-h-64 overflow-y-auto pr-1 scrollbar-hide">
                      {Object.entries(EMOJI_LIBRARY).map(([category, emojis]) => (
                        <div key={category}>
                          <p className="text-[9px] text-zinc-600 uppercase font-bold tracking-tighter mb-2">{category}</p>
                          <div className="grid grid-cols-6 gap-1">
                            {emojis.map(emoji => (
                              <button 
                                key={emoji}
                                onClick={() => {
                                  sendReaction(emoji);
                                  setIsEmojiPickerOpen(false);
                                }}
                                className="w-8 h-8 flex items-center justify-center text-lg hover:bg-white/10 rounded-lg transition-all hover:scale-110"
                              >
                                {emoji}
                              </button>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </div>
      </motion.div>

      {/* SIDEBAR ANALYTICS */}
      <AnimatePresence mode="wait">
        {!isFocusMode && (
          <motion.aside 
            key="sidebar"
            initial={{ width: 0, opacity: 0, marginLeft: -16 }}
            animate={{ width: 336, opacity: 1, marginLeft: 0 }}
            exit={{ width: 0, opacity: 0, marginLeft: -16 }}
            transition={{ duration: 0.5, ease: [0.4, 0, 0.2, 1] }}
            className="w-[336px] flex flex-col gap-4 overflow-hidden"
          >
            {/* TAB SWITCHER */}
            <div className="flex bg-zinc-900/50 p-1 rounded-2xl border border-white/5 mx-1">
              <button 
                onClick={() => setSidebarTab('insights')}
                className={cn(
                  "flex-1 py-2 px-1 rounded-xl text-[9px] font-bold uppercase tracking-widest transition-all",
                  sidebarTab === 'insights' ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-lg shadow-cyan-900/20" : "text-zinc-500 hover:text-zinc-300"
                )}
              >
                Telemetry
              </button>
              <button 
                onClick={() => setSidebarTab('chat')}
                className={cn(
                  "flex-1 py-2 px-1 rounded-xl text-[9px] font-bold uppercase tracking-widest transition-all relative overflow-hidden",
                  sidebarTab === 'chat' ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-lg shadow-cyan-900/20" : "text-zinc-500 hover:text-zinc-300"
                )}
              >
                Comms
                {sidebarTab !== 'chat' && <div className="absolute top-1 right-2 w-1.5 h-1.5 rounded-full bg-cyan-500 animate-pulse" />}
              </button>
              <button 
                onClick={() => setSidebarTab('profile')}
                className={cn(
                  "flex-1 py-2 px-1 rounded-xl text-[9px] font-bold uppercase tracking-widest transition-all",
                  sidebarTab === 'profile' ? "bg-cyan-500/10 text-cyan-400 border border-cyan-500/20 shadow-lg shadow-cyan-900/20" : "text-zinc-500 hover:text-zinc-300"
                )}
              >
                Profile
              </button>
            </div>

            <AnimatePresence mode="wait">
              {sidebarTab === 'insights' ? (
                <motion.div 
                  key="insights-tab"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="flex-1 flex flex-col gap-4 overflow-hidden"
                >
                  {/* EMOTIONAL TRENDS CARD */}
                  <div className="flex-1 immersive-card p-6 flex flex-col">
                    <div className="flex items-center justify-between mb-8">
                      <div>
                        <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400">Live Sentiment</h2>
                        <p className="text-[10px] font-mono text-zinc-600 mt-1 uppercase">Stream Analysis 4x29</p>
                      </div>
                      <div className="flex items-center gap-2">
                        <button 
                          onClick={() => setIsAnalysisPaused(!isAnalysisPaused)}
                          className={cn(
                            "p-1.5 rounded-lg border transition-all",
                            isAnalysisPaused 
                              ? "bg-amber-500/10 border-amber-500/30 text-amber-500 hover:bg-amber-500/20" 
                              : "bg-white/5 border-white/10 text-zinc-500 hover:text-white"
                          )}
                          title={isAnalysisPaused ? "Resume Analysis" : "Pause Analysis"}
                        >
                          {isAnalysisPaused ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
                        </button>
                        <button 
                          onClick={resetAnalysis}
                          className="p-1.5 rounded-lg border bg-white/5 border-white/10 text-zinc-500 hover:text-white hover:bg-white/10 transition-all"
                          title="Reset Analysis"
                        >
                          <RotateCcw className="w-3.5 h-3.5" />
                        </button>
                        <div className={cn(
                          "px-2 py-1 text-[10px] rounded border uppercase animate-pulse",
                          isAnalysisPaused 
                            ? "bg-amber-500/10 text-amber-500 border-amber-500/20" 
                            : "bg-cyan-500/10 text-cyan-400 border-cyan-500/20"
                        )}>
                          {isAnalysisPaused ? "Paused" : "Running"}
                        </div>
                      </div>
                    </div>
                    
                    {/* CHART VISUALIZATION */}
                    <div className="flex-1 min-h-[200px] px-2">
                      <ResponsiveContainer width="100%" height="100%">
                        <AreaChart data={[...history].reverse()}>
                          <defs>
                            <linearGradient id="colorEngagement" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#22d3ee" stopOpacity={0}/>
                            </linearGradient>
                            <linearGradient id="colorConfidence" x1="0" y1="0" x2="0" y2="1">
                              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3}/>
                              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0}/>
                            </linearGradient>
                          </defs>
                          <CartesianGrid strokeDasharray="3 3" stroke="#ffffff10" vertical={false} />
                          <XAxis dataKey="timestamp" hide={true} />
                          <YAxis hide={true} domain={[0, 100]} />
                          <Tooltip 
                            contentStyle={{ backgroundColor: '#18181b', border: '1px solid #ffffff10', borderRadius: '8px', fontSize: '10px' }}
                            itemStyle={{ color: '#22d3ee' }}
                          />
                          <Area type="monotone" dataKey="engagement" stroke="#22d3ee" fillOpacity={1} fill="url(#colorEngagement)" strokeWidth={2} isAnimationActive={false} />
                          <Area type="monotone" dataKey={(v) => v.confidence * 100} name="Confidence" stroke="#3b82f6" fillOpacity={1} fill="url(#colorConfidence)" strokeWidth={1} strokeDasharray="4 4" isAnimationActive={false} />
                        </AreaChart>
                      </ResponsiveContainer>
                    </div>

                    <div className="mt-8 space-y-4">
                      <div className="p-4 bg-white/5 rounded-2xl border border-white/10">
                        <p className="text-[10px] text-slate-500 uppercase font-bold mb-1">AI Recommendation</p>
                        <div className="text-sm italic text-slate-300 leading-tight">
                          <ReactMarkdown>{emotion.insight || "Collecting baseline data..."}</ReactMarkdown>
                        </div>
                      </div>

                      <div className="space-y-3 pt-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-cyan-400"></div> Engagement</span>
                          <span className="font-mono text-cyan-400">{emotion.engagement}%</span>
                        </div>
                        <div className="flex items-center justify-between text-xs opacity-50">
                          <span className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-blue-400"></div> System Status</span>
                          <span className="font-mono">OPTIMAL</span>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* PARTICIPANTS LIST */}
                  <div className="immersive-card p-6 h-56 overflow-y-auto scrollbar-hide">
                     <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400 mb-6">In Call ({participants.length})</h2>
                     <div className="space-y-4">
                       {participants.map((p) => {
                         const currentStatus = p.isLocal ? (emotion.primary || "Neutral") : p.status;
                         const colorClass = getEmotionColor(currentStatus);
                         const ringClass = getEmotionRingColor(currentStatus);
                         
                         return (
                           <div key={p.id} className={cn("flex items-center gap-3", p.isLocal ? "" : "opacity-80")}>
                             <div className={cn(
                               "w-10 h-10 rounded-full border border-white/10 ring-2 transition-all duration-700 overflow-hidden bg-zinc-800",
                               ringClass
                             )}>
                                <img src={getAvatarUrl(p.avatarSeed, p.isLocal, avatarConfig)} alt={p.name} referrerPolicy="no-referrer" className="w-full h-full object-cover" />
                             </div>
                             <div className="flex-1">
                               <p className="text-sm font-medium leading-none">{p.name}</p>
                               <p className={cn(
                                 "text-[10px] uppercase font-mono mt-1 tracking-tighter transition-colors duration-700",
                                 colorClass
                               )}>
                                 {currentStatus}
                               </p>
                             </div>
                           </div>
                         );
                       })}
                     </div>
                  </div>
                </motion.div>
              ) : sidebarTab === 'chat' ? (
                <motion.div 
                  key="chat-tab"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="flex-1 immersive-card flex flex-col overflow-hidden"
                >
                  <div className="p-6 border-b border-white/5">
                    <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400">Secure Comms</h2>
                    <p className="text-[10px] font-mono text-zinc-600 mt-1 uppercase">End-to-End Encrypted</p>
                  </div>
                  
                  <div className="flex-1 overflow-y-auto p-4 space-y-4 scrollbar-hide">
                    {messages.map((msg, i) => {
                      const participant = participants.find(p => p.id === msg.senderId);
                      return (
                        <motion.div 
                          key={msg.id}
                          initial={{ opacity: 0, y: 10 }}
                          animate={{ opacity: 1, y: 0 }}
                          transition={{ delay: i * 0.01 }}
                          className={cn(
                            "flex gap-3",
                            participant?.isLocal ? "flex-row-reverse" : ""
                          )}
                        >
                          <div className="w-8 h-8 rounded-full border border-white/10 bg-zinc-800 flex-shrink-0">
                            <img 
                              src={getAvatarUrl(participant?.avatarSeed || 'default', participant?.isLocal, avatarConfig)} 
                              alt="Avatar" 
                              className="w-full h-full rounded-full"
                            />
                          </div>
                          <div className={cn(
                            "max-w-[80%] p-3 rounded-2xl text-xs leading-relaxed relative",
                            participant?.isLocal ? "bg-cyan-500/20 text-cyan-50 border border-cyan-500/30 font-medium" : "bg-white/5 text-slate-300 border border-white/5"
                          )}>
                            {msg.text}
                            <div className="flex items-center justify-between mt-1 gap-2">
                              <span className="text-[9px] opacity-40 font-mono">
                                {msg.timestamp}
                              </span>
                              {participant?.isLocal && (
                                <span className="text-cyan-400">
                                  {msg.status === 'read' ? <CheckCheck className="w-3 h-3" /> : <Check className={cn("w-3 h-3", msg.status === 'delivered' ? "opacity-100" : "opacity-40")} />}
                                </span>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      );
                    })}
                    {thinkingParticipants.length > 0 && (
                      <motion.div 
                        initial={{ opacity: 0 }} 
                        animate={{ opacity: 1 }}
                        className="flex gap-3 mb-2"
                      >
                        <div className="w-8 h-8 rounded-full border border-cyan-500/30 bg-cyan-500/5 flex items-center justify-center relative">
                          <div className="absolute inset-0 rounded-full border border-cyan-500/20 animate-ping" />
                          <Brain className="w-4 h-4 text-cyan-400 animate-pulse" />
                        </div>
                        <div className="bg-cyan-500/5 text-[9px] p-2 rounded-xl text-cyan-400/60 uppercase font-bold tracking-widest italic flex items-center gap-2">
                          {thinkingParticipants.map(id => participants.find(p => p.id === id)?.name || 'Guest').join(', ')} is processing...
                          <span className="flex gap-0.5">
                            <span className="w-0.5 h-0.5 bg-cyan-400 rounded-full animate-bounce [animation-delay:0s]" />
                            <span className="w-0.5 h-0.5 bg-cyan-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                            <span className="w-0.5 h-0.5 bg-cyan-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                          </span>
                        </div>
                      </motion.div>
                    )}

                    {typingParticipants.length > 0 && (
                      <motion.div 
                        initial={{ opacity: 0 }} 
                        animate={{ opacity: 1 }}
                        className="flex gap-3"
                      >
                        <div className="w-8 h-8 rounded-full border border-white/10 bg-zinc-800 flex items-center justify-center">
                          <span className="w-1 h-1 bg-cyan-400 rounded-full animate-bounce" />
                          <span className="w-1 h-1 bg-cyan-400 rounded-full animate-bounce [animation-delay:0.2s] mx-0.5" />
                          <span className="w-1 h-1 bg-cyan-400 rounded-full animate-bounce [animation-delay:0.4s]" />
                        </div>
                        <div className="bg-white/5 text-[9px] p-2 rounded-xl text-zinc-500 uppercase font-bold tracking-widest italic">
                          {(() => {
                            const names = typingParticipants.map(id => participants.find(p => p.id === id)?.name || 'Someone');
                            if (names.length === 1) return `${names[0]} is typing...`;
                            if (names.length === 2) return `${names[0]} and ${names[1]} are typing...`;
                            return `${names[0]} and ${names.length - 1} others are typing...`;
                          })()}
                        </div>
                      </motion.div>
                    )}
                  </div>

                  <div className="p-4 bg-black/20 border-t border-white/5">
                    <form 
                      onSubmit={(e) => {
                        e.preventDefault();
                        if (!newMessage.trim()) return;
                        const msg: ChatMessage = {
                          id: Date.now().toString(),
                          senderId: 'local',
                          text: newMessage,
                          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
                          status: 'sent'
                        };
                        setMessages([...messages, msg]);
                        setNewMessage("");
                      }}
                      className="flex gap-2"
                    >
                      <input 
                        type="text" 
                        value={newMessage}
                        onChange={(e) => setNewMessage(e.target.value)}
                        placeholder="Transmit data..."
                        className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-xs focus:outline-none focus:ring-1 ring-cyan-500/50 transition-all font-mono"
                      />
                      <button className="p-2 bg-cyan-500/20 text-cyan-400 rounded-xl border border-cyan-500/30 hover:bg-cyan-500/30 transition-all">
                        <Zap className="w-4 h-4" />
                      </button>
                    </form>
                  </div>
                </motion.div>
              ) : (
                <motion.div 
                  key="profile-tab"
                  initial={{ opacity: 0, x: 20 }}
                  animate={{ opacity: 1, x: 0 }}
                  exit={{ opacity: 0, x: -20 }}
                  className="flex-1 immersive-card flex flex-col overflow-hidden"
                >
                  <div className="p-6 border-b border-white/5">
                    <h2 className="text-xs font-bold uppercase tracking-widest text-slate-400">Persona Editor</h2>
                    <p className="text-[10px] font-mono text-zinc-600 mt-1 uppercase">Modify Digital Identity</p>
                  </div>
                  
                  <div className="flex-1 overflow-y-auto p-6 space-y-8 scrollbar-hide">
                    <div className="flex flex-col items-center">
                      <div className="w-32 h-32 rounded-full border-4 border-cyan-500/20 shadow-2xl bg-zinc-800 overflow-hidden mb-4 relative">
                         <img src={getAvatarUrl('local', true, avatarConfig)} alt="Preview" className="w-full h-full" />
                      </div>
                      <div className="flex items-center gap-2">
                        <div className="w-2 h-2 rounded-full bg-cyan-500 animate-pulse" />
                        <span className="text-[9px] font-mono text-cyan-400 uppercase tracking-widest">Neural Link Syncing</span>
                      </div>
                    </div>

                    <div className="space-y-6">
                      {/* TOP / HAIR */}
                      <div className="space-y-3">
                        <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center justify-between">
                          Hairstyle <span className="text-cyan-400/60 lowercase">{avatarConfig.top}</span>
                        </label>
                        <div className="flex gap-2 overflow-x-auto scrollbar-hide pb-2">
                          {AVATAR_OPTIONS.top.map(opt => (
                            <button 
                              key={opt}
                              onClick={() => setAvatarConfig(prev => ({ ...prev, top: opt }))}
                              className={cn(
                                "flex-shrink-0 w-12 h-12 rounded-xl border transition-all overflow-hidden bg-zinc-900 p-1",
                                avatarConfig.top === opt ? "border-cyan-500 shadow-lg shadow-cyan-900/20" : "border-white/5 hover:border-white/20"
                              )}
                            >
                              <img src={`https://api.dicebear.com/7.x/avataaars/svg?top=${opt}`} alt={opt} className="w-full h-full" />
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* ACCESSORIES */}
                      <div className="space-y-3">
                        <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center justify-between">
                          Eyewear <span className="text-cyan-400/60 lowercase">{avatarConfig.accessories}</span>
                        </label>
                        <div className="flex gap-2 overflow-x-auto scrollbar-hide pb-2">
                          {AVATAR_OPTIONS.accessories.map(opt => (
                            <button 
                              key={opt}
                              onClick={() => setAvatarConfig(prev => ({ ...prev, accessories: opt }))}
                              className={cn(
                                "flex-shrink-0 w-12 h-12 rounded-xl border transition-all overflow-hidden bg-zinc-900 p-1",
                                avatarConfig.accessories === opt ? "border-cyan-500 shadow-lg shadow-cyan-900/20" : "border-white/5 hover:border-white/20"
                              )}
                            >
                              <img src={`https://api.dicebear.com/7.x/avataaars/svg?accessories=${opt}`} alt={opt} className="w-full h-full" />
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* CLOTHES */}
                      <div className="space-y-3">
                        <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center justify-between">
                          Clothing <span className="text-cyan-400/60 lowercase">{avatarConfig.clothes}</span>
                        </label>
                        <div className="flex gap-2 overflow-x-auto scrollbar-hide pb-2">
                          {AVATAR_OPTIONS.clothes.map(opt => (
                            <button 
                              key={opt}
                              onClick={() => setAvatarConfig(prev => ({ ...prev, clothes: opt }))}
                              className={cn(
                                "flex-shrink-0 w-12 h-12 rounded-xl border transition-all overflow-hidden bg-zinc-900 p-1",
                                avatarConfig.clothes === opt ? "border-cyan-500 shadow-lg shadow-cyan-900/20" : "border-white/5 hover:border-white/20"
                              )}
                            >
                              <img src={`https://api.dicebear.com/7.x/avataaars/svg?clothes=${opt}`} alt={opt} className="w-full h-full" />
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* EYES */}
                      <div className="space-y-3">
                        <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center justify-between">
                          Neural Optics (Eye Shapes) <span className="text-cyan-400/60 lowercase">{avatarConfig.eyes}</span>
                        </label>
                        <div className="grid grid-cols-4 gap-2">
                          {AVATAR_OPTIONS.eyes.map(opt => (
                            <button 
                              key={opt}
                              onClick={() => setAvatarConfig(prev => ({ ...prev, eyes: opt }))}
                              className={cn(
                                "aspect-square rounded-xl border transition-all overflow-hidden bg-zinc-900 p-1 flex items-center justify-center",
                                avatarConfig.eyes === opt ? "border-cyan-500 bg-cyan-500/10 shadow-lg shadow-cyan-900/20" : "border-white/5 hover:border-white/20"
                              )}
                              title={opt}
                            >
                              <img 
                                src={`https://api.dicebear.com/7.x/avataaars/svg?eyes=${opt}&baseColor=transparent&top=Blank&mouth=Default&accessories=Blank&clothes=Blank`} 
                                alt={opt} 
                                className="w-10 h-10 object-contain" 
                              />
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* MOUTH */}
                      <div className="space-y-3">
                        <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest flex items-center justify-between">
                          Expression Module (Mouth Expressions) <span className="text-cyan-400/60 lowercase">{avatarConfig.mouth}</span>
                        </label>
                        <div className="grid grid-cols-4 gap-2">
                          {AVATAR_OPTIONS.mouth.map(opt => (
                            <button 
                              key={opt}
                              onClick={() => setAvatarConfig(prev => ({ ...prev, mouth: opt }))}
                              className={cn(
                                "aspect-square rounded-xl border transition-all overflow-hidden bg-zinc-900 p-1 flex items-center justify-center",
                                avatarConfig.mouth === opt ? "border-cyan-500 bg-cyan-500/10 shadow-lg shadow-cyan-900/20" : "border-white/5 hover:border-white/20"
                              )}
                              title={opt}
                            >
                              <img 
                                src={`https://api.dicebear.com/7.x/avataaars/svg?mouth=${opt}&baseColor=transparent&top=Blank&eyes=Default&accessories=Blank&clothes=Blank`} 
                                alt={opt} 
                                className="w-10 h-10 object-contain" 
                              />
                            </button>
                          ))}
                        </div>
                      </div>

                      {/* BACKGROUND EFFECTS */}
                      <div className="space-y-4 pt-4 border-t border-white/5">
                        <label className="text-[9px] font-bold text-zinc-500 uppercase tracking-widest">Environment Modulation</label>
                        
                        <div className="grid grid-cols-3 gap-2">
                          <button 
                            onClick={() => setBackgroundEffect('none')}
                            className={cn(
                              "py-2 rounded-xl border text-[10px] font-bold uppercase transition-all",
                              backgroundEffect === 'none' ? "bg-cyan-500/10 border-cyan-500/50 text-cyan-400" : "bg-white/5 border-white/5 text-zinc-500 hover:border-white/20"
                            )}
                          >
                            Raw Link
                          </button>
                          <button 
                            onClick={() => setBackgroundEffect('blur')}
                            className={cn(
                              "py-2 rounded-xl border text-[10px] font-bold uppercase transition-all",
                              backgroundEffect === 'blur' ? "bg-cyan-500/10 border-cyan-500/50 text-cyan-400" : "bg-white/5 border-white/5 text-zinc-500 hover:border-white/20"
                            )}
                          >
                            Blur
                          </button>
                          <button 
                            onClick={() => setBackgroundEffect('image')}
                            className={cn(
                              "py-2 rounded-xl border text-[10px] font-bold uppercase transition-all",
                              backgroundEffect === 'image' ? "bg-cyan-500/10 border-cyan-500/50 text-cyan-400" : "bg-white/5 border-white/5 text-zinc-500 hover:border-white/20"
                            )}
                          >
                            Virtual
                          </button>
                        </div>

                        {backgroundEffect === 'image' && (
                          <div className="flex gap-2 overflow-x-auto scrollbar-hide pb-2">
                            {VIRTUAL_BACKGROUNDS.map(bg => (
                              <button 
                                key={bg.id}
                                onClick={() => setSelectedBackground(bg.url)}
                                className={cn(
                                  "flex-shrink-0 w-20 aspect-video rounded-lg border transition-all overflow-hidden relative group",
                                  selectedBackground === bg.url ? "border-cyan-500 ring-2 ring-cyan-500/20" : "border-white/5 hover:border-white/20"
                                )}
                              >
                                <img src={bg.url} alt={bg.name} className="w-full h-full object-cover" />
                                <div className="absolute inset-0 bg-black/40 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center">
                                  <span className="text-[8px] font-bold text-white uppercase">{bg.name}</span>
                                </div>
                              </button>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </motion.aside>
        )}
      </AnimatePresence>

      <canvas ref={canvasRef} className="hidden" />
    </div>
  );
}

