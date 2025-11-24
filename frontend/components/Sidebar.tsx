'use client';

import { useState } from 'react';
import Image from 'next/image';
import { Home, Upload, BarChart2, Settings, Menu, X } from 'lucide-react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const Sidebar = () => {
    const [isOpen, setIsOpen] = useState(true);
    const pathname = usePathname();

    return (
        <>
            {/* Toggle Button */}
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="fixed top-4 left-4 z-50 p-2 bg-[#242324] border border-[#2C2C2B] rounded-md hover:bg-[#2C2C2B] transition-colors"
            >
                {isOpen ? <X size={20} /> : <Menu size={20} />}
            </button>

            {/* Sidebar */}
            <div
                className={`fixed left-0 top-0 h-screen bg-[#191818] border-r border-[#2C2C2B] flex flex-col transition-transform duration-300 z-40 ${isOpen ? 'translate-x-0' : '-translate-x-full'
                    } w-64`}
            >
                {/* Logo - Outside sidebar visually but inside the component */}
                <div className="p-6 pt-16 border-b border-[#2C2C2B]">
                    <Image
                        src="/logo_best.png"
                        alt="GR Logo"
                        width={150}
                        height={40}
                        className="w-full h-auto"
                    />
                </div>

                {/* Navigation */}
                <nav className="flex-1 p-4 space-y-1">
                    <NavItem
                        href="/"
                        icon={<Home size={20} />}
                        label="Home"
                        isActive={pathname === '/'}
                    />
                    <NavItem
                        href="/upload"
                        icon={<Upload size={20} />}
                        label="Data Upload"
                        isActive={pathname === '/upload'}
                    />
                    <NavItem
                        href="/dashboard"
                        icon={<BarChart2 size={20} />}
                        label="Dashboard"
                        isActive={pathname === '/dashboard'}
                    />
                    <NavItem
                        href="/settings"
                        icon={<Settings size={20} />}
                        label="Settings"
                        isActive={pathname === '/settings'}
                    />
                </nav>

                {/* Footer */}
                <div className="p-4 border-t border-[#2C2C2B]">
                    <div className="text-xs text-gray-500">
                        Gazoo Analyst v2.0
                    </div>
                </div>
            </div>

            {/* Overlay for mobile */}
            {isOpen && (
                <div
                    className="fixed inset-0 bg-black/50 z-30 lg:hidden"
                    onClick={() => setIsOpen(false)}
                />
            )}
        </>
    );
};

const NavItem = ({
    href,
    icon,
    label,
    isActive
}: {
    href: string;
    icon: React.ReactNode;
    label: string;
    isActive: boolean;
}) => {
    return (
        <Link
            href={href}
            className={`flex items-center gap-3 px-4 py-2 rounded-md transition-colors ${isActive
                ? 'bg-[#0276CF] text-[#fafafa]'
                : 'text-gray-400 hover:text-white hover:bg-[#242324]'
                }`}
        >
            {icon}
            <span>{label}</span>
        </Link>
    );
};

export default Sidebar;
