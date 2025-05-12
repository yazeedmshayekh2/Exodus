import React, { useState } from 'react';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid, 
  Card, 
  CardContent,
  List,
  ListItem, 
  ListItemIcon,
  ListItemText, 
  Button,
  Tab,
  Tabs,
  useTheme
} from '@mui/material';
import { 
  CheckCircle as CheckIcon,
  Language as LanguageIcon,
  Search as SearchIcon,
  Code as CodeIcon,
  Storage as StorageIcon,
  Devices as DevicesIcon
} from '@mui/icons-material';
import { motion } from 'framer-motion';
import { Link as RouterLink } from 'react-router-dom';
import Lottie from 'lottie-react';
import robotAnimation from '../assets/robot-animation.json';

const MotionPaper = motion(Paper);
const MotionCard = motion(Card);

const AboutPage = () => {
  const theme = useTheme();
  const [language, setLanguage] = useState('en');
  const [tab, setTab] = useState(0);
  const isRtl = language === 'ar';

  const handleLanguageChange = () => {
    setLanguage(prev => prev === 'en' ? 'ar' : 'en');
  };

  const handleTabChange = (event, newValue) => {
    setTab(newValue);
  };

  const technologies = [
    { 
      icon: <CodeIcon />, 
      name_en: 'FastAPI', 
      name_ar: 'فاست أي بي آي',
      description_en: 'A modern, high-performance web framework for building APIs', 
      description_ar: 'إطار عمل حديث عالي الأداء لبناء واجهات برمجة التطبيقات'
    },
    { 
      icon: <SearchIcon />, 
      name_en: 'Sentence Transformers', 
      name_ar: 'محولات الجمل',
      description_en: 'For semantic search and finding similar questions', 
      description_ar: 'للبحث الدلالي والعثور على الأسئلة المتشابهة'
    },
    { 
      icon: <StorageIcon />, 
      name_en: 'Qdrant Vector Database', 
      name_ar: 'قاعدة بيانات Qdrant المتجهية',
      description_en: 'For efficient similarity matching', 
      description_ar: 'للمطابقة المتشابهة الفعالة' 
    },
    { 
      icon: <LanguageIcon />, 
      name_en: 'Multilingual Support', 
      name_ar: 'دعم متعدد اللغات',
      description_en: 'Handles both English and Arabic queries', 
      description_ar: 'يعالج الاستعلامات باللغتين الإنجليزية والعربية'
    },
    { 
      icon: <DevicesIcon />, 
      name_en: 'React Frontend', 
      name_ar: 'واجهة React الأمامية',
      description_en: 'Modern, responsive user interface', 
      description_ar: 'واجهة مستخدم حديثة ومتجاوبة'
    }
  ];

  const features = [
    { 
      text_en: 'Understands the meaning behind your questions', 
      text_ar: 'يفهم المعنى وراء أسئلتك'
    },
    { 
      text_en: 'Provides accurate answers from the knowledge base', 
      text_ar: 'يقدم إجابات دقيقة من قاعدة المعرفة'
    },
    { 
      text_en: 'Switches automatically between English and Arabic', 
      text_ar: 'ينتقل تلقائيًا بين اللغتين الإنجليزية والعربية'
    },
    { 
      text_en: 'Uses AI to generate responses for new questions', 
      text_ar: 'يستخدم الذكاء الاصطناعي لإنشاء إجابات للأسئلة الجديدة'
    },
    { 
      text_en: 'Continuously improves with more usage', 
      text_ar: 'يتحسن باستمرار مع المزيد من الاستخدام'
    }
  ];

  return (
    <Box sx={{ direction: isRtl ? 'rtl' : 'ltr' }}>
      <Box sx={{ mb: 3, display: 'flex', justifyContent: 'flex-end' }}>
        <Button 
          variant="outlined" 
          startIcon={<LanguageIcon />}
          onClick={handleLanguageChange}
        >
          {isRtl ? 'English' : 'عربي'}
        </Button>
      </Box>

      <MotionPaper
        elevation={3}
        sx={{ p: 4, borderRadius: 2, mb: 4 }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <Grid container spacing={4} alignItems="center">
          <Grid item xs={12} md={7}>
            <Typography variant="h3" component="h1" gutterBottom>
              {isRtl ? 'حول روبوت الدردشة للأسئلة الشائعة' : 'About the FAQ Chatbot'}
            </Typography>
            <Typography variant="h6" color="text.secondary" gutterBottom>
              {isRtl 
                ? 'مساعد ذكي يمكنه الإجابة على الأسئلة المتكررة باللغتين الإنجليزية والعربية'
                : 'A smart assistant that can answer frequently asked questions in both English and Arabic'}
            </Typography>
            <Typography variant="body1" paragraph>
              {isRtl
                ? 'يستخدم روبوت الدردشة هذا تقنية البحث الدلالي للعثور على الإجابات الأكثر صلة من قاعدة المعرفة، مما يضمن حصولك على معلومات دقيقة بسرعة.'
                : 'This chatbot uses semantic search technology to find the most relevant answers from a knowledge base, ensuring you get precise information quickly.'}
            </Typography>
            <Button 
              variant="contained" 
              size="large" 
              component={RouterLink} 
              to="/"
              sx={{ mt: 2 }}
            >
              {isRtl ? 'جرّب الدردشة الآن' : 'Try chatting now'}
            </Button>
          </Grid>
          <Grid item xs={12} md={5}>
            <Box sx={{ maxWidth: 300, mx: 'auto' }}>
              <Lottie animationData={robotAnimation} loop />
            </Box>
          </Grid>
        </Grid>
      </MotionPaper>

      <Typography variant="h4" component="h2" gutterBottom sx={{ mt: 6, mb: 3 }}>
        {isRtl ? 'الميزات الرئيسية' : 'Key Features'}
      </Typography>
      
      <Grid container spacing={3}>
        {features.map((feature, index) => (
          <Grid item xs={12} sm={6} md={4} key={index}>
            <MotionCard
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.3, delay: index * 0.1 }}
              sx={{ height: '100%' }}
            >
              <CardContent sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                <CheckIcon color="success" fontSize="large" />
                <Typography variant="body1">
                  {isRtl ? feature.text_ar : feature.text_en}
                </Typography>
              </CardContent>
            </MotionCard>
          </Grid>
        ))}
      </Grid>

      <Typography variant="h4" component="h2" gutterBottom sx={{ mt: 6, mb: 3 }}>
        {isRtl ? 'التقنيات المستخدمة' : 'Technologies Used'}
      </Typography>

      <Paper sx={{ mb: 4 }}>
        <Tabs
          value={tab}
          onChange={handleTabChange}
          variant="scrollable"
          scrollButtons="auto"
          sx={{ borderBottom: 1, borderColor: 'divider' }}
        >
          {technologies.map((tech, index) => (
            <Tab 
              key={index} 
              label={isRtl ? tech.name_ar : tech.name_en} 
              icon={tech.icon} 
              iconPosition="start"
            />
          ))}
        </Tabs>
        
        {technologies.map((tech, index) => (
          <Box
            key={index}
            role="tabpanel"
            hidden={tab !== index}
            id={`tech-tabpanel-${index}`}
            sx={{ p: 3 }}
          >
            {tab === index && (
              <Typography variant="body1">
                {isRtl ? tech.description_ar : tech.description_en}
              </Typography>
            )}
          </Box>
        ))}
      </Paper>
    </Box>
  );
};

export default AboutPage; 