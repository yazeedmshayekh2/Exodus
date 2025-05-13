import React, { useState } from 'react';
import { 
  AppBar, 
  Box, 
  Toolbar, 
  Typography, 
  Button, 
  Container, 
  IconButton, 
  Drawer, 
  List, 
  ListItem, 
  ListItemIcon, 
  ListItemText, 
  useMediaQuery, 
  useTheme 
} from '@mui/material';
import { 
  Menu as MenuIcon, 
  Close as CloseIcon, 
  ChatBubble as ChatIcon, 
  Info as InfoIcon, 
  BugReport as DebugIcon,
  Language as LanguageIcon
} from '@mui/icons-material';
import { Link as RouterLink, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';

const MotionBox = motion(Box);

const Layout = ({ children }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const [drawer, setDrawer] = useState(false);
  const [language, setLanguage] = useState('en'); // 'en' or 'ar'
  const location = useLocation();

  const toggleDrawer = () => {
    setDrawer(!drawer);
  };

  const toggleLanguage = () => {
    setLanguage(language === 'en' ? 'ar' : 'en');
  };

  const isRtl = language === 'ar';

  const navItems = [
    { text: isRtl ? 'الدردشة' : 'Chat', path: '/', icon: <ChatIcon /> },
    { text: isRtl ? 'حول' : 'About', path: '/about', icon: <InfoIcon /> },
    { text: isRtl ? 'تصحيح الأخطاء' : 'Debug', path: '/debug', icon: <DebugIcon /> },
  ];

  return (
    <Box sx={{ 
      display: 'flex', 
      flexDirection: 'column', 
      minHeight: '100vh',
      direction: isRtl ? 'rtl' : 'ltr'
    }}>
      <AppBar position="static" color="primary">
        <Toolbar>
          {isMobile && (
            <IconButton
              color="inherit"
              aria-label="open drawer"
              edge="start"
              onClick={toggleDrawer}
              sx={{ mr: 2 }}
            >
              <MenuIcon />
            </IconButton>
          )}
          
          <Typography 
            variant="h6" 
            component="div" 
            sx={{ 
              flexGrow: 1,
              fontWeight: 'bold',
              textAlign: isMobile ? 'center' : 'left'
            }}
          >
            {isRtl ? 'روبوت الدردشة ثنائي اللغة' : 'Bilingual FAQ Chatbot'}
          </Typography>
          
          {!isMobile && navItems.map((item) => (
            <Button
              key={item.path}
              component={RouterLink}
              to={item.path}
              color="inherit"
              sx={{ 
                mx: 1,
                fontWeight: location.pathname === item.path ? 'bold' : 'normal',
                borderBottom: location.pathname === item.path 
                  ? '2px solid white' 
                  : '2px solid transparent'
              }}
              startIcon={item.icon}
            >
              {item.text}
            </Button>
          ))}
          
          <IconButton color="inherit" onClick={toggleLanguage}>
            <LanguageIcon />
          </IconButton>
        </Toolbar>
      </AppBar>
      
      <Drawer
        anchor={isRtl ? 'right' : 'left'}
        open={drawer}
        onClose={toggleDrawer}
      >
        <Box
          sx={{ width: 250 }}
          role="presentation"
          onClick={toggleDrawer}
        >
          <Box sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            p: 2,
            bgcolor: 'primary.main',
            color: 'white'
          }}>
            <Typography variant="h6">
              {isRtl ? 'القائمة' : 'Menu'}
            </Typography>
            <IconButton color="inherit" onClick={toggleDrawer}>
              <CloseIcon />
            </IconButton>
          </Box>
          
          <List>
            {navItems.map((item) => (
              <ListItem 
                button 
                key={item.path} 
                component={RouterLink} 
                to={item.path}
                selected={location.pathname === item.path}
                sx={{
                  bgcolor: location.pathname === item.path ? 'action.selected' : 'transparent',
                  borderLeft: !isRtl && location.pathname === item.path 
                    ? `4px solid ${theme.palette.primary.main}` 
                    : '4px solid transparent',
                  borderRight: isRtl && location.pathname === item.path 
                    ? `4px solid ${theme.palette.primary.main}` 
                    : '4px solid transparent',
                }}
              >
                <ListItemIcon>
                  {item.icon}
                </ListItemIcon>
                <ListItemText primary={item.text} />
              </ListItem>
            ))}
          </List>
        </Box>
      </Drawer>
      
      <MotionBox
        component="main"
        sx={{
          flexGrow: 1,
          py: 3,
          px: 2,
          bgcolor: 'background.default'
        }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Container maxWidth="lg">
          {children}
        </Container>
      </MotionBox>
      
      <Box
        component="footer"
        sx={{
          py: 2,
          px: 2,
          mt: 'auto',
          backgroundColor: theme.palette.grey[100],
          textAlign: 'center'
        }}
      >
        <Typography variant="body2" color="text.secondary">
          © 2025 {isRtl ? 'روبوت الدردشة ثنائي اللغة' : 'Bilingual FAQ Chatbot'} | 
          {isRtl ? ' بُني بـ ❤️ بواسطة Basel' : ' Built with ❤️ by Yazeed'}
        </Typography>
      </Box>
    </Box>
  );
};

export default Layout; 