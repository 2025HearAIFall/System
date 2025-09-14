import React, { useState } from 'react';

const emojiMap = {
  happiness: '😊 (기쁨)',
  sadness: '😢 (슬픔)',
  angry: '😠 (화남)',
  fear: '😱 (공포)',
  surprise: '😲 (놀람)',
  neutral: '😐 (중립)',
  disgust: '🤢 (혐오)',
};

export default function EmotionSubtitleUI() {
  const [text, setText] = useState('');
  const [emotion, setEmotion] = useState('');
  const [loading, setLoading] = useState(false);
  const [file, setFile] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleSubmit = async () => {
    if (!file) return;
    setLoading(true);
  
    const formData = new FormData();
    formData.append("file", file);
  
    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });
  
      const data = await response.json();
      console.log("[DEBUG] 서버 응답:", data);  // 🔍 확인용 로그
  
      const eng = data.emotion.split(" ")[0];  // "surprise (놀람)" → "surprise"
      setText(data.text);
      setEmotion(eng);  // emojiMap에 들어갈 key만 사용
    } catch (error) {
      console.error("예측 실패:", error);
    } finally {
      setLoading(false);
    }
  };
  

  return (
    <div style={{ textAlign: 'center', paddingTop: '50px' }}>
      <h1>Handbridge</h1>

      <div style={{ border: '2px solid teal', borderRadius: '16px', height: '200px', margin: '20px auto', maxWidth: '600px', display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
        {loading ? (
          <p style={{ fontSize: '20px' }}>분석 중...</p>
        ) : (
          <>
            <p style={{ fontSize: '24px' }}>{text}</p>
            <p style={{ fontSize: '40px' }}>{emojiMap[emotion]}</p>
          </>
        )}
      </div>

      <input type="file" accept=".wav" onChange={handleFileChange} />
      <br />
      <button onClick={handleSubmit} style={{ marginTop: '10px', padding: '10px 20px', fontSize: '16px', backgroundColor: '#007BFF', color: 'white', border: 'none', borderRadius: '8px' }}>
        분석하기
      </button>
    </div>
  );
}
