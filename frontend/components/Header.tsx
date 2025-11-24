'use client';

import Image from 'next/image';
import Link from 'next/link';
import { usePathname } from 'next/navigation';

const tabs = [
    { name: 'Overview', href: '/' },
    { name: 'Data Upload', href: '/upload' },
    { name: 'Dashboard', href: '/dashboard' },
    { name: 'Settings', href: '/settings' },
];

const Header = () => {
    const pathname = usePathname();

    return (
        <header className="border-b border-[#2C2C2B] bg-[#191818]">
            <div className="px-6 py-3">
                {/* Logo */}
                <div className="flex items-center mb-3">
                    <Image
                        src="/logo_best.png"
                        alt="GR Logo"
                        width={120}
                        height={32}
                        className="h-8 w-auto"
                    />
                </div>

                {/* Navigation Tabs - Full Width Container */}
                <nav className="flex items-center w-full h-fit">
                    {tabs.map((tab) => {
                        const isActive = pathname === tab.href;
                        return (
                            <Link
                                key={tab.name}
                                href={tab.href}
                                className={`text-sm mx-2 text-neutral-500 hover:bg-white/5 transition-all ease-in-out duration-150 cursor-pointer h-10 flex items-center px-3 rounded-md relative ${isActive ? 'bg-white/10 text-white' : ''
                                    }`}
                            >
                                {tab.name}
                                {isActive && (
                                    <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-white rounded-full" />
                                )}
                            </Link>
                        );
                    })}
                </nav>
            </div>
        </header>
    );
};

export default Header;
