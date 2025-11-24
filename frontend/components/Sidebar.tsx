import Image from 'next/image';
import { Home, Upload, BarChart2, Settings } from 'lucide-react';
import Link from 'next/link';

const Sidebar = () => {
    return (
        <div className="h-screen w-64 bg-[#191818] border-r border-[#2C2C2B] flex flex-col">
            <div className="p-6 border-b border-[#2C2C2B]">
                <Image src="/logo.png" alt="GR Logo" width={150} height={40} className="w-full h-auto" />
            </div>

            <nav className="flex-1 p-4 space-y-2">
                <NavItem href="/" icon={<Home size={20} />} label="Home" />
                <NavItem href="/upload" icon={<Upload size={20} />} label="Data Upload" />
                <NavItem href="/dashboard" icon={<BarChart2 size={20} />} label="Dashboard" />
                <NavItem href="/settings" icon={<Settings size={20} />} label="Settings" />
            </nav>

            <div className="p-4 border-t border-[#2C2C2B]">
                <div className="text-xs text-gray-500">
                    Gazoo Analyst v2.0
                </div>
            </div>
        </div>
    );
};

const NavItem = ({ href, icon, label }: { href: string; icon: React.ReactNode; label: string }) => {
    return (
        <Link href={href} className="flex items-center gap-3 px-4 py-3 text-gray-400 hover:text-white hover:bg-[#242324] rounded-md transition-colors">
            {icon}
            <span>{label}</span>
        </Link>
    );
};

export default Sidebar;
