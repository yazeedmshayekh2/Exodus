import React, { useState, useRef, useEffect } from 'react';
import { 
  Box, 
  TextField, 
  Button, 
  Paper, 
  Typography, 
  CircularProgress,
  IconButton,
  Snackbar,
  Alert
} from '@mui/material';
import { Send as SendIcon } from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
import ReactMarkdown from 'react-markdown';
import rehypeSanitize from 'rehype-sanitize';
import rehypeRaw from 'rehype-raw';
import chatService from '../api/chatService';
import ModelSelector from './ModelSelector';

const MotionPaper = motion(Paper);

const ChatInterface = ({ language, setLanguage }) => {
  const [messages, setMessages] = useState([
    {
      role: 'bot',
      content: language === 'ar' 
        ? 'مرحباً! كيف يمكنني مساعدتك اليوم؟' 
        : 'Hello! How can I help you today?',
      language: language
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'info' });
  const messagesEndRef = useRef(null);
  const isRtl = language === 'ar';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (input.trim() === '') return;

    try {
      // Add user message
      const userMessage = {
        role: 'user',
        content: input,
        language: language,
      };

      setMessages(prev => [...prev, userMessage]);
      setInput('');
      setIsLoading(true);

      // Get response from API
      const response = await chatService.sendQuery(input.trim());
      
      // Update language if response comes back in a different language
      if (response.language !== language) {
        setLanguage(response.language);
      }

      // Add bot message
      const botMessage = {
        role: 'bot',
        content: response.response,
        language: response.language,
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      
      // Add error message
      const errorMessage = {
        role: 'bot',
        content: language === 'ar'
          ? 'عذراً، حدث خطأ أثناء معالجة طلبك. يرجى المحاولة مرة أخرى.'
          : 'Sorry, there was an error processing your request. Please try again.',
        language: language,
        isError: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleModelChange = (model) => {
    // Show notification when model changes
    setNotification({
      open: true,
      message: language === 'ar'
        ? `تم تغيير النموذج إلى ${model.name}`
        : `Changed model to ${model.name}`,
      severity: 'success'
    });
    
    // Add system message
    const systemMessage = {
      role: 'bot',
      content: language === 'ar'
        ? `*تم تغيير النموذج إلى ${model.name}*`
        : `*Changed model to ${model.name}*`,
      language: language,
      isSystem: true
    };
    
    setMessages(prev => [...prev, systemMessage]);
  };

  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  return (
    <Box sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Model selector */}
      <ModelSelector onModelChange={handleModelChange} language={language} />
      
      <Paper 
        elevation={3}
        sx={{ 
          flex: 1, 
          mb: 2, 
          p: 2, 
          overflow: 'auto',
          maxHeight: 'calc(100vh - 250px)',
          minHeight: '400px',
          bgcolor: '#f8f9fa',
          direction: isRtl ? 'rtl' : 'ltr'
        }}
      >
        <AnimatePresence>
          {messages.map((message, index) => (
            <MotionPaper
              key={index}
              elevation={1}
              sx={{
                maxWidth: '80%',
                mb: 2,
                p: 2,
                bgcolor: message.role === 'user' ? '#e3f2fd' : 
                        message.isSystem ? '#f0f4c3' : '#ffffff',
                borderRadius: '10px',
                alignSelf: message.role === 'user' ? 'flex-end' : 'flex-start',
                ml: message.role === 'user' ? 'auto' : 0,
                mr: message.role === 'user' ? 0 : 'auto',
                borderTopRightRadius: message.role === 'user' ? 0 : '10px',
                borderTopLeftRadius: message.role === 'user' ? '10px' : 0,
                position: 'relative',
                ...(message.isError && { bgcolor: '#ffebee' })
              }}
              initial={{ opacity: 0, y: 20, scale: 0.8 }}
              animate={{ opacity: 1, y: 0, scale: 1 }}
              transition={{ duration: 0.3 }}
            >
              <Typography 
                variant="body1" 
                sx={{ whiteSpace: 'pre-wrap', textAlign: isRtl ? 'right' : 'left' }}
              >
                <ReactMarkdown 
                  rehypePlugins={[rehypeSanitize, rehypeRaw]}
                >
                  {message.content}
                </ReactMarkdown>
              </Typography>
            </MotionPaper>
          ))}
          {isLoading && (
            <MotionPaper
              elevation={1}
              sx={{
                maxWidth: '80%',
                mb: 2,
                p: 2,
                bgcolor: '#f5f5f5',
                borderRadius: '10px',
                borderTopLeftRadius: 0,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
              }}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.2 }}
            >
              <CircularProgress size={20} thickness={5} sx={{ mr: 1 }} />
              <Typography variant="body2" color="textSecondary">
                {isRtl ? 'جاري التفكير...' : 'Thinking...'}
              </Typography>
            </MotionPaper>
          )}
        </AnimatePresence>
        <div ref={messagesEndRef} />
      </Paper>
      
      <Box sx={{ display: 'flex', alignItems: 'center', direction: isRtl ? 'rtl' : 'ltr' }}>
        <TextField
          fullWidth
          variant="outlined"
          placeholder={isRtl ? 'اكتب سؤالك هنا...' : 'Type your question...'}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isLoading}
          sx={{ 
            direction: isRtl ? 'rtl' : 'ltr',
            '& .MuiOutlinedInput-root': {
              borderRadius: '25px',
              pr: 1
            }
          }}
          InputProps={{
            endAdornment: (
              <IconButton 
                color="primary" 
                onClick={handleSend} 
                disabled={isLoading || input.trim() === ''}
                sx={{ ml: 1 }}
              >
                {isLoading ? <CircularProgress size={24} /> : <SendIcon />}
              </IconButton>
            )
          }}
        />
      </Box>
      
      <Snackbar 
        open={notification.open} 
        autoHideDuration={3000} 
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity} 
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Box>
  );
};

export default ChatInterface; 