import React, { useState } from 'react';
import { Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import ChatPage from './pages/ChatPage';
import AboutPage from './pages/AboutPage';
import DebugPage from './pages/DebugPage';
import { Box, Typography, Button } from '@mui/material';
import { Link as RouterLink } from 'react-router-dom';

// 404 Page
const NotFound = () => (
  <Box 
    sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      alignItems: 'center', 
      justifyContent: 'center',
      minHeight: '50vh',
      textAlign: 'center',
      p: 3
    }}
  >
    <Typography variant="h1" component="h1" sx={{ mb: 2, fontSize: { xs: '5rem', md: '8rem' } }}>
      404
    </Typography>
    <Typography variant="h4" component="h2" sx={{ mb: 4 }}>
      Page Not Found
    </Typography>
    <Typography variant="body1" color="text.secondary" sx={{ mb: 4, maxWidth: 500 }}>
      The page you are looking for doesn't exist or has been moved.
    </Typography>
    <Button 
      variant="contained" 
      component={RouterLink} 
      to="/"
      size="large"
    >
      Back to Home
    </Button>
  </Box>
);

const App = () => {
  const [language, setLanguage] = useState('en');

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<ChatPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route path="/debug" element={<DebugPage />} />
        <Route path="*" element={<NotFound />} />
      </Routes>
    </Layout>
  );
};

export default App; 