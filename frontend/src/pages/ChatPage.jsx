import React, { useState } from 'react';
import { Box, Typography, Paper, Tab, Tabs } from '@mui/material';
import ChatInterface from '../components/ChatInterface';
import { motion } from 'framer-motion';
import Lottie from 'lottie-react';
import chatAnimation from '../assets/chat-animation.json';

const MotionBox = motion(Box);

const ChatPage = () => {
  const [language, setLanguage] = useState('en');
  const isRtl = language === 'ar';

  return (
    <MotionBox
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.5 }}
      sx={{ direction: isRtl ? 'rtl' : 'ltr' }}
    >
      <Box sx={{ 
        display: 'flex', 
        flexDirection: { xs: 'column', md: 'row' },
        alignItems: 'center',
        gap: 4,
        mb: 4
      }}>
        <Box sx={{ 
          width: { xs: '100%', md: '40%' },
          textAlign: { xs: 'center', md: isRtl ? 'right' : 'left' }
        }}>
          <Typography variant="h3" component="h1" gutterBottom>
            {isRtl ? 'روبوت الدردشة للأسئلة الشائعة' : 'FAQ Chatbot'}
          </Typography>
          <Typography variant="h5" color="text.secondary" gutterBottom>
            {isRtl 
              ? 'يمكنني الإجابة على أسئلتك باللغتين العربية والإنجليزية' 
              : 'I can answer your questions in English and Arabic'}
          </Typography>
          <Typography variant="body1" color="text.secondary">
            {isRtl
              ? 'ما عليك سوى كتابة سؤالك وسأبذل قصارى جهدي للمساعدة!'
              : 'Just type your question and I will do my best to help!'}
          </Typography>
        </Box>

        <Box sx={{ 
          width: { xs: '60%', md: '30%' },
          display: { xs: 'none', md: 'block' }
        }}>
          <Lottie animationData={chatAnimation} loop />
        </Box>
      </Box>

      <Paper 
        elevation={3}
        sx={{ 
          p: 3,
          borderRadius: 2,
          background: 'white'
        }}
      >
        <ChatInterface language={language} setLanguage={setLanguage} />
      </Paper>
    </MotionBox>
  );
};

export default ChatPage; 