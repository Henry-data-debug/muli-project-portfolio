
import React from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Calendar, Clock, ArrowRight, User } from 'lucide-react';

const Blog = () => {
  const blogPosts = [
    {
      title: "10 Power Automate Triggers That Will Transform Your Workflow",
      excerpt: "Discover the most powerful and underutilized triggers in Power Automate that can revolutionize how your business handles routine tasks and processes.",
      category: "Power Automate",
      readTime: "8 min read",
      date: "Dec 15, 2024",
      image: "https://images.unsplash.com/photo-1461749280684-dccba630e2f6?q=80&w=400&h=250&fit=crop",
      featured: true
    },
    {
      title: "Excel Formulas Every Business Analyst Should Master",
      excerpt: "Advanced Excel techniques including dynamic arrays, XLOOKUP, and Power Query that will make you 10x more efficient at data analysis.",
      category: "Excel",
      readTime: "12 min read",
      date: "Dec 10, 2024",
      image: "https://images.unsplash.com/photo-1518770660439-4636190af475?q=80&w=400&h=250&fit=crop"
    },
    {
      title: "WhatsApp Business API: Complete Integration Guide",
      excerpt: "Step-by-step tutorial on integrating WhatsApp Business API with your existing systems for automated customer communication.",
      category: "API Integration",
      readTime: "15 min read",
      date: "Dec 5, 2024",
      image: "https://images.unsplash.com/photo-1488590528505-98d2b5aba04b?q=80&w=400&h=250&fit=crop"
    },
    {
      title: "SharePoint Automation: From Beginner to Expert",
      excerpt: "Complete guide to automating SharePoint workflows, from simple approvals to complex multi-stage processes with conditional logic.",
      category: "SharePoint",
      readTime: "10 min read",
      date: "Nov 28, 2024",
      image: "https://images.unsplash.com/photo-1487058792275-0ad4aaf24ca7?q=80&w=400&h=250&fit=crop"
    },
    {
      title: "Building Real-Time Dashboards That Actually Work",
      excerpt: "Best practices for creating Power BI dashboards that provide actionable insights and drive business decisions in real-time.",
      category: "Power BI",
      readTime: "11 min read",
      date: "Nov 20, 2024",
      image: "https://images.unsplash.com/photo-1498050108023-c5249f4df085?q=80&w=400&h=250&fit=crop"
    },
    {
      title: "API Integration Strategies for Small Businesses",
      excerpt: "Practical approaches to connecting multiple business systems without breaking the bank or requiring extensive technical expertise.",
      category: "Automation",
      readTime: "9 min read",
      date: "Nov 15, 2024",
      image: "https://images.unsplash.com/photo-1531297484001-80022131f5a1?q=80&w=400&h=250&fit=crop"
    }
  ];

  const categories = ["All", "Power Automate", "Excel", "SharePoint", "Power BI", "API Integration", "Automation"];

  return (
    <section id="blog" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-4xl font-bold text-gray-900 mb-4">
            Blog & Tips
          </h2>
          <p className="text-xl text-gray-600 max-w-3xl mx-auto">
            Insights, tutorials, and best practices for workflow automation, data analysis, and business process optimization.
          </p>
        </div>

        {/* Category Filter */}
        <div className="flex flex-wrap justify-center gap-2 mb-12">
          {categories.map((category, index) => (
            <Button
              key={index}
              variant={index === 0 ? "default" : "outline"}
              size="sm"
              className={index === 0 ? "bg-blue-600 hover:bg-blue-700" : ""}
            >
              {category}
            </Button>
          ))}
        </div>

        {/* Featured Post */}
        <div className="mb-12">
          <Card className="overflow-hidden border-0 shadow-xl">
            <div className="grid grid-cols-1 lg:grid-cols-2">
              <div className="relative">
                <img 
                  src={blogPosts[0].image} 
                  alt={blogPosts[0].title}
                  className="w-full h-64 lg:h-full object-cover"
                />
                <Badge className="absolute top-4 left-4 bg-blue-600">
                  Featured
                </Badge>
              </div>
              <div className="p-8 flex flex-col justify-center">
                <div className="flex items-center gap-4 text-sm text-gray-600 mb-4">
                  <Badge variant="outline">{blogPosts[0].category}</Badge>
                  <div className="flex items-center gap-1">
                    <Calendar className="h-4 w-4" />
                    {blogPosts[0].date}
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    {blogPosts[0].readTime}
                  </div>
                </div>
                <h3 className="text-2xl font-bold text-gray-900 mb-4 leading-tight">
                  {blogPosts[0].title}
                </h3>
                <p className="text-gray-600 mb-6 leading-relaxed">
                  {blogPosts[0].excerpt}
                </p>
                <Button className="bg-blue-600 hover:bg-blue-700 w-fit">
                  Read Full Article
                  <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </div>
          </Card>
        </div>

        {/* Regular Posts Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
          {blogPosts.slice(1).map((post, index) => (
            <Card key={index} className="overflow-hidden border-0 shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
              <div className="relative">
                <img 
                  src={post.image} 
                  alt={post.title}
                  className="w-full h-48 object-cover"
                />
                <Badge variant="secondary" className="absolute top-4 left-4 bg-white/90 text-gray-800">
                  {post.category}
                </Badge>
              </div>
              
              <CardContent className="p-6">
                <div className="flex items-center gap-3 text-sm text-gray-600 mb-3">
                  <div className="flex items-center gap-1">
                    <Calendar className="h-4 w-4" />
                    {post.date}
                  </div>
                  <div className="flex items-center gap-1">
                    <Clock className="h-4 w-4" />
                    {post.readTime}
                  </div>
                </div>
                
                <h3 className="font-bold text-gray-900 mb-3 leading-tight line-clamp-2">
                  {post.title}
                </h3>
                
                <p className="text-gray-600 text-sm mb-4 leading-relaxed line-clamp-3">
                  {post.excerpt}
                </p>
                
                <Button variant="outline" size="sm" className="w-full group">
                  Read More
                  <ArrowRight className="ml-2 h-4 w-4 group-hover:translate-x-1 transition-transform" />
                </Button>
              </CardContent>
            </Card>
          ))}
        </div>

        <div className="mt-16 text-center">
          <Button size="lg" variant="outline" className="border-blue-600 text-blue-600 hover:bg-blue-600 hover:text-white">
            View All Articles
            <ArrowRight className="ml-2 h-5 w-5" />
          </Button>
        </div>
      </div>
    </section>
  );
};

export default Blog;
